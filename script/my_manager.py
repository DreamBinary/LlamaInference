# -*- coding:utf-8 -*-
# @FileName : my_manager.py
# @Time : 2024/1/9 19:56
# @Author :fiv
import asyncio

import uvloop
from lightllm.server.httpserver.manager import HttpServerManager, ReqStatus

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class MyHttpServerManager(HttpServerManager):
    async def generate(self, prompt, sampling_params, request_id, multimodal_params):
        # enable_multimodal ==>> False
        # if self.enable_multimodal:
        #     assert (
        #             len(multimodal_params.images) <= self.args.cache_capacity
        #     ), "too many images!"
        #     await self._alloc_multimodal_resources(multimodal_params)
        #     prompt_ids = self.tokenizer.encode(prompt, multimodal_params)
        # else:
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)

        # if prompt_tokens > self.max_req_input_len:
        #     # use long_truncation_mode to truncate long input len req.
        #     if self.args.long_truncation_mode is None:
        #         raise ValueError(
        #             f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}"
        #         )
        #     elif self.args.long_truncation_mode == "head":
        #         prompt_ids = prompt_ids[-self.max_req_input_len:]
        #         prompt_tokens = len(prompt_ids)
        #     elif self.args.long_truncation_mode == "center":
        #         prompt_ids = prompt_ids[0:self.max_req_input_len // 2] + prompt_ids[-(
        #                     self.max_req_input_len - self.max_req_input_len // 2):]
        #         prompt_tokens = len(prompt_ids)
        #         assert prompt_tokens == self.max_req_input_len
        #     else:
        #         assert False, "error args"

        # req_total_len = prompt_tokens + sampling_params.max_new_tokens
        # if req_total_len > self.max_req_total_len:
        #     raise ValueError(
        #         f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
        #     )
        # if req_total_len + 1 > self.total_token_num:
        #     raise ValueError(
        #         f"the req token total len + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}"
        #     )

        # sampling_params.stop_sentences_to_token_ids(self.tokenizer)

        req_status = ReqStatus(request_id, multimodal_params)
        event = req_status.event
        self.req_id_to_out_inf[request_id] = req_status

        # 寻找是否有可用的prompt cache 可用
        prompt_cache_len, prompt_cache_req_id = self._find_prompt_cache_req(prompt_ids)

        # if self.enable_multimodal:
        #     self.send_to_visual.send_pyobj(
        #         (prompt_ids, sampling_params, multimodal_params, request_id, prompt_cache_len, prompt_cache_req_id))
        # else:
        #     self.send_to_router.send_pyobj(
        #         (prompt_ids, sampling_params, multimodal_params, request_id, prompt_cache_len, prompt_cache_req_id))
        self.send_to_router.send_pyobj(
            (prompt_ids, sampling_params, multimodal_params, request_id, prompt_cache_len, prompt_cache_req_id))
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

            async with req_status.lock:
                event.clear()
                if len(req_status.out_token_info_list) == 0:
                    continue

                for out_str, metadata, finish_status in req_status.out_token_info_list:
                    metadata["prompt_tokens"] = prompt_tokens
                    yield out_str, metadata, finish_status

                    if finish_status.is_finished():
                        try:
                            del self.req_id_to_out_inf[request_id]
                            await self._release_multimodal_resources(multimodal_params)
                        except:
                            pass
                        return
                req_status.out_token_info_list.clear()
        return
