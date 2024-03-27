# # -*- coding: utf-8 -*-
# """
# @File    : log_util.py
# @Author  : fiv
# @Time    : 2023/12/29 22:17
# """
#
import logging
import os
import pathlib
import time


def get_log(filename=None) -> logging.Logger:
    log = logging.getLogger(__name__)
    dir_path = pathlib.Path(__file__).parent.parent
    if filename is None:
        filename = "log"
    t = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    filename = f"{filename}_{t}.log"
    file_path = os.path.join(dir_path, "log", filename)

    logging.basicConfig(
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s:  %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename=file_path, encoding="utf-8"),
        ],
        level=logging.INFO
    )
    log.info(f"Log file path: {file_path}")
    return log
