# -*- coding:utf-8 -*-
# @FileName : async_data.py
# @Time : 2024/1/8 20:21
# @Author :fiv
from typing import List


# class AsyncData:
#     def __init__(self, data: List):
#         self._queue = asyncio.Queue()
#         for i, item in enumerate(data):
#             self._queue.put_nowait((i, item))
#         self._queue.put_nowait(StopIteration)
#
#     def __aiter__(self):
#         return self
#
#     async def __anext__(self):
#         result = await self._queue.get()
#         if result is StopIteration:
#             raise StopAsyncIteration
#         return result


class AsyncData:
    def __init__(self, data: List):
        self.cnt = -1
        self.length = len(data)
        self.data = [(i, item) for i, item in enumerate(data)]

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.cnt += 1
        if self.cnt >= self.length:
            raise StopAsyncIteration
        return self.data[self.cnt]
