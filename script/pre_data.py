# -*- coding:utf-8 -*-
# @FileName : pre_data.py
# @Time : 2024/1/1 13:19
# @Author :fiv

import json
from typing import Tuple, List, Dict

from tqdm import tqdm


class PreDataset:
    def __init__(self, data_path: str = None, num_samples: int = None):
        if data_path is None:
            data_path = "../data/scrambled_sampled_dataset.json"
        with open(data_path) as f:
            requests = json.load(f)
        if num_samples is not None:
            requests = requests[0:num_samples]
        data = {}
        max_output_len = -1
        max_input_len = -1
        for prompt, input_len, output_len in tqdm(requests, desc="Preprocessing dataset"):
            if output_len not in data:
                data[output_len] = []
            max_output_len = max(max_output_len, output_len)
            max_input_len = max(max_input_len, input_len)
            data[output_len].append(prompt)
        self.data = data
        self.length = len(requests)
        self.max_output_len = max_output_len
        self.max_input_len = max_input_len

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.length

    def items(self):
        return self.data.items()

    def get_max_len(self):
        return self.max_input_len, self.max_output_len


def get_class_data(dataset: str, num_samples: int = None) -> Dict[int, List[str]]:
    data = get_full_data(dataset, num_samples)
    data = {output_len: [p for p, _, o in data if o == output_len] for _, _, output_len in data}
    return data


def get_data(dataset: str, num_samples: int = None) -> List[Tuple[int, str, int]]:
    data = get_full_data(dataset, num_samples)
    data = [(index, prompt, output_len) for index, prompt, _, output_len in data]
    return data


def get_sort_data(dataset: str, num_samples: int = None) -> List[Tuple[int, str, int]]:
    data = get_sort_full_data(dataset, num_samples)
    data = [(index, prompt, output_len) for index, prompt, _, output_len in data]
    return data


def get_sort_full_data(dataset: str, num_samples: int = None) -> List[Tuple[int, str, int, int]]:
    data = get_full_data(dataset, num_samples)
    data = sorted(data, key=lambda x: -x[3])
    return data


def get_full_data(dataset: str, num_samples: int = None) -> List[Tuple[int, str, int, int]]:
    assert dataset is not None, "dataset is None"
    with open(dataset) as f:
        data = json.load(f)
    if num_samples is not None:
        data = data[0:num_samples]
    data = [(t, p, i, o) for t, (p, i, o) in enumerate(data)]
    return data


def get_my_data(dataset: str, num_samples: int = None) -> List[Tuple[int, str, int]]:
    data = get_full_data(dataset)
    result = []
    data = sorted(data, key=lambda x: x[2])  # sort by input_len
    result.extend(data[0: 1000])
    data = data[1000:]
    data = sorted(data, key=lambda x: -x[3])  # sort by output_len
    result.extend(data)
    result = [(t, p, o) for t, p, _, o in result]
    return result

# def
