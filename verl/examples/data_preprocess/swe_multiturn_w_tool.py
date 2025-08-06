# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a simple multi-turn dataset for testing
"""

import argparse
import os
import json
import pandas as pd

import datasets
from tqdm import tqdm

def make_map_fn(split):
    def process_fn(example, idx):
        # message = example['func_loc_message']
        # instance_id = example['instance_id']
        # loc_gt = example['loc_gt']
        message = example.pop("func_loc_message")
        instance_id = example.pop("instance_id")
        repo = example.pop("repo")
        base_commit = example.pop("base_commit")
        loc_gt = example.pop("loc_gt")
        message[0]['content'] = message[0]['content'].split("Function calls you can use are as follows:")[0]
        message[1]['content'] = message[1]['content'].split("Your response should follow the format as follows:")[0].replace("<action>","<tool_call>").replace("</action>","</tool_call>")
        data = {
            "data_source": "swe_loc",
            "prompt": message,
            "ability": "func_loc",
            "reward_model": {"style": "rule", "ground_truth": loc_gt},
            "extra_info": {
                "split": split,
                "index": idx,
                "ground_truth": loc_gt,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "get_code_of_class": {
                        "create_kwargs": {"swe_instance_id":instance_id,"repo":repo,"base_commit":base_commit,"ground_truth": loc_gt},
                        # "execute_kwargs": {},
                        # "calc_reward_kwargs": {},
                        # "release_kwargs": {},
                    },
                    "get_code_of_file_function": {
                        "create_kwargs": {"swe_instance_id":instance_id,"repo":repo,"base_commit":base_commit,"ground_truth": loc_gt},
                        # "execute_kwargs": {},
                        # "calc_reward_kwargs": {},
                        # "release_kwargs": {},
                    },
                    "get_code_of_class_function": {
                        "create_kwargs": {"swe_instance_id":instance_id,"repo":repo,"base_commit":base_commit,"ground_truth": loc_gt},
                        # "execute_kwargs": {},
                        # "calc_reward_kwargs": {},
                        # "release_kwargs": {},
                    },
                    "exit": {
                        "create_kwargs": {"swe_instance_id":instance_id,"repo":repo,"base_commit":base_commit,"ground_truth": loc_gt},
                        # "execute_kwargs": {},
                        # "calc_reward_kwargs": {},
                        # "release_kwargs": {},
                    }
                },
            },
        }
        return data

    def debug_precess_fn(example, idx):
        result = process_fn(example, idx)
        print(result)
        breakpoint()
        return result

    return process_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_files", default="")
    parser.add_argument("--test_file", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    # dataset = datasets.load_dataset("json", data_files=args.loc_files)[:500]
    
    # vis_keys = None
    # with open(args.loc_files,"r") as f:
    #     for line in tqdm(f):
    #         item = json.loads(line)
    #         if not vis_keys:
    #             vis_keys = item.keys()
    #         elif not vis_keys == item.keys():
    #             breakpoint()

    def preprocess_item(item):
        processed = {}
        for key in item:
            value = item[key]
            # 统一将 null/None 转为空字符串 ""，或保留 null 但指定 allow_null=True
            if value is None:
                processed[key] = ""  # 或者改为其他默认值
            else:
                processed[key] = value
        return processed

    def load_and_preprocess(file):
        dataset = []
        # 加载并预处理数据
        with open(file, "r") as f:
            for line in tqdm(f):
                item = json.loads(line)
                # item = preprocess_item(item)
                if item['loc_gt']:
                    # item['loc_gt'] = [item['loc_gt']]
                    # item.pop("loc_gt")
                    item["loc_gt"] = str(item["loc_gt"])
                    dataset.append(item)
        return dataset

    # 创建数据集
    train_dataset = load_and_preprocess(args.loc_files)
    print(f"train_dataset size: {len(train_dataset)}")
    breakpoint()
    # print(train_dataset[0]['loc_gt'])
    # breakpoint()
    train_dataset = datasets.Dataset.from_list(train_dataset)
    # print(train_dataset[0]['loc_gt'])
    # breakpoint()

    train_dataset_mapped = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    test_dataset = load_and_preprocess(args.test_file)
    test_dataset = datasets.Dataset.from_list(test_dataset)
    test_dataset_mapped = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset_mapped.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    test_dataset_mapped.to_parquet(os.path.join(args.output_dir, "test.parquet"))


if __name__ == "__main__":
    main()
