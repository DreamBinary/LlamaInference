# -*- coding:utf-8 -*-
# @FileName : my_mii_pipe.py
# @Time : 2024/1/6 13:12
# @Author :fiv
import queue
import threading
import time
from typing import Dict, Tuple, List, Any

from mii.batching.ragged_batching import RaggedBatchBase
from tqdm import tqdm


class MyMIIPipeline(RaggedBatchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tid = threading.get_ident()
        self.log: List[Tuple[float, int]] = []

    def __call__(self, inputs: List[Tuple[int, str, int, int]]) -> Tuple[
        float, List[int], List[int], List[Tuple[float, int]]]:
        input_num_tokens = []
        output_num_tokens = []
        cnt = len(inputs)

        pbar = tqdm(total=cnt, desc="OUTPUT")

        start = time.perf_counter()

        for index, prompt, prompt_len, output_len in inputs:
            request_kwargs = {
                "max_length": output_len + prompt_len,
                "ignore_eos": False,
                "top_p": 1.0,
                "temperature": 1.0,
                "do_sample": False,
            }
            self._put_request(index, prompt, request_kwargs)

        self.schedule_requests()

        if self.is_rank_0:
            # Rank 0 runs generate() until all responses are returned
            while cnt > 0:
                self.generate()
                while not self.result_queues[self.tid].empty():
                    uid, prompt_length, generated_length = self._get_response()
                    input_num_tokens.append(prompt_length)
                    output_num_tokens.append(generated_length + prompt_length)
                    pbar.update(1)
                    self._queue_flush_request(uid)
                    cnt -= 1
            # Ensure final flush requests broadcast and
            # kick ranks 1 -> n out of the while loop
            self._bcast_requests(force=True)
        else:
            # Ranks 1 -> n just run generate() until there are no more requests
            while self.scheduled_requests:
                self.generate()

        end = time.perf_counter()
        pbar.close()
        return end - start, input_num_tokens, output_num_tokens, self.log

    def _put_request(self, uid: int, input: str, kwargs: Dict[str, Any]) -> None:
        self.result_queues[self.tid] = queue.Queue()
        input_tokens = self.tokenizer.encode(input)
        request = self.make_request(self.tid, uid, input_tokens, kwargs)
        self.request_queue.put(request)

    def _get_response(self) -> Tuple[int, int, int]:
        result = self.result_queues[self.tid].get()
        # Include the decoding time.
        self.tokenizer.decode(result[1])
        # result[0] -->> uid
        # result[1] -->> generated_tokens
        # result[2] -->> prompt_length
        # result[3] -->> generated_length
        # result[4] -->> finish_reason
        return result[0], result[2], result[3]

    def generate(self) -> None:
        # 1. Get a batch of requests, broadcast to all ranks
        scheduled_requests = self._bcast_requests()

        # 2. Flush for uids that are finished generating
        self.flush(scheduled_requests.requests_to_flush.uids)

        # 3. Put new tokens into inference engine
        if scheduled_requests.requests_to_run:
            next_token_logits = self.put(
                scheduled_requests.requests_to_run.uids,
                scheduled_requests.requests_to_run.tokens,
            )

        # short circuit if not rank 0, only rank 0 does scheduling and postprocessing of logits
        if not self.is_rank_0:
            return

        # 4. Launch logit processing and token generation
        running_requests = scheduled_requests.requests_to_run
        running_requests.update_seq_length()
        if running_requests:
            next_tokens, done_tokens = self._process_logits(
                next_token_logits, running_requests
            )
            running_requests.next_tokens = next_tokens
            running_requests.done_tokens = done_tokens

        # 5. Schedule requests while we wait for the forward pass to finish
        self._reset_scheduler_bookkeeping()

        # 6. Accumulate generated tokens, check completion, and generate output
        for r in running_requests.last_in_prompt:
            r.accumulate_generated_token()
            self._num_generated_tokens += 1
            if r.stop_generation or r.stream:
                self._generate_output(r)
            if not r.stop_generation:
                r.set_next_as_input()
                self.request_queue.put(r)

        # 7. Update scheduled requests
        self.scheduled_requests.prune(running_requests.completed.uids)
        self.schedule_requests()

        self.write_log()

    def write_log(self):
        self.log.append((time.monotonic(), self._num_generated_tokens))
        self._num_generated_tokens = 0
