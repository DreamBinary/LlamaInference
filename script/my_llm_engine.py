# -*- coding:utf-8 -*-
# @FileName : my_llm_engine.py
# @Time : 2024/1/7 20:17
# @Author :fiv

import asyncio
import time
from functools import partial
from typing import (Any, List, Optional, Tuple, Type, Dict)

from ray.util.placement_group import PlacementGroup
from vllm import AsyncLLMEngine
from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig
from vllm.config import ModelConfig
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput


class MyLLMEngine(LLMEngine):

    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, parallel_config: ParallelConfig,
                 scheduler_config: SchedulerConfig, placement_group: Optional["PlacementGroup"],
                 log_stats: bool) -> None:
        super().__init__(model_config, cache_config, parallel_config, scheduler_config, placement_group, log_stats)
        self.log: List[Tuple[float, int, int, int, int, int]] = []

    # timestamp, prompt_num_tokens, generation_num_tokens, num_running, num_swapped, num_pending

    # def step(self) -> None:
    #     seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
    #     if scheduler_outputs.is_empty():
    #         return
    #
    #     output = self._run_workers(
    #         "execute_model",
    #         seq_group_metadata_list=seq_group_metadata_list,
    #         blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
    #         blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
    #         blocks_to_copy=scheduler_outputs.blocks_to_copy,
    #     )
    #
    #     step_outputs = self._process_model_outputs(output, scheduler_outputs)
    #     step_outputs = filter(lambda x: x.finished, step_outputs)
    #     self.outputs.extend(step_outputs)

    def _log_system_stats(self, prompt_run: bool, num_batched_tokens: int) -> None:
        if prompt_run:
            self.log.append((time.monotonic(), num_batched_tokens, 0, len(self.scheduler.running),
                             len(self.scheduler.swapped), len(self.scheduler.waiting)))
        else:
            self.log.append((time.monotonic(), 0, num_batched_tokens, len(self.scheduler.running),
                             len(self.scheduler.swapped), len(self.scheduler.waiting)))


class _AsyncLLMEngine(MyLLMEngine):
    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if not scheduler_outputs.is_empty():
            # Execute the model.
            all_outputs = await self._run_workers_async(
                "execute_model",
                driver_kwargs={
                    "seq_group_metadata_list": seq_group_metadata_list,
                    "blocks_to_swap_in": scheduler_outputs.blocks_to_swap_in,
                    "blocks_to_swap_out": scheduler_outputs.blocks_to_swap_out,
                    "blocks_to_copy": scheduler_outputs.blocks_to_copy,
                })

            # Only the driver worker returns the sampling results.
            output = all_outputs[0]
        else:
            output = []

        return self._process_model_outputs(output, scheduler_outputs)

    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        coros = []

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Run the driver worker asynchronously.
        driver_executor = getattr(self.driver_worker, method)
        coros.append(asyncio.get_event_loop().run_in_executor(
            None, partial(driver_executor, *driver_args, **driver_kwargs)))

        # Run the ray workers asynchronously.
        for worker in self.workers:
            coros.append(worker.execute_method.remote(method, *args, **kwargs))

        all_outputs = await asyncio.gather(*coros)
        return all_outputs


class MyAsyncLLMEngine(AsyncLLMEngine):
    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine
