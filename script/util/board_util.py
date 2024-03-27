# # -*- coding:utf-8 -*-
# # @FileName : board_util.py
# # @Time : 2024/1/7 13:43
# # @Author :fiv
# import os
# import pathlib
# import time
# from typing import List, Dict
#
# from torch.utils.tensorboard import SummaryWriter
#
#
# def draw_tensorboard(filename=None, data: Dict[str, List[float]] = None):
#     dir_path = pathlib.Path(__file__).parent.parent
#     if filename is None:
#         filename = "tensorboard"
#     t = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
#     filename = f"{filename}_{t}"
#     log_dir = os.path.join(dir_path, "log", "run", filename)
#     writer = SummaryWriter(log_dir=log_dir)
#     for key, value in data.items():
#         for i, v in enumerate(value):
#             writer.add_scalar(key, v, i, new_style=True)
#     writer.close()
