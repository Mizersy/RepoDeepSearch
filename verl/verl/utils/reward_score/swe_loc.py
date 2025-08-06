# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from verl.repoSearcher.util.postprocess_data import extract_code_blocks

def extract_locs(locs):
    current_file_name = None
    results = {}
    for loc in locs:
        for line in loc.splitlines():
            if line.strip().endswith(".py"):
                current_file_name = line.strip()
            elif line.strip() and any(
                line.startswith(w)
                for w in ["line:", "function:", "class:", "variable:"]
            ):
                if current_file_name not in results:
                    results[current_file_name] = []
                if line not in results[current_file_name]: # deduplicate
                    results[current_file_name].append(line)

    # return {fn: ["\n".join(results[fn])] for fn in results.keys()}
    return results

def get_ndcg_reward(y_pred, y_true_set, k=5):
    """
    计算 nDCG@k 作为 RL 的 reward。
    y_pred: 模型的预测列表
    y_true_set: 正确答案的集合 (set)
    k: 考虑的 top-k 位置
    """
    import numpy as np
    y_pred_k = y_pred[:k]
    
    # 计算 DCG@k
    dcg = 0.0
    for i, item in enumerate(y_pred_k):
        if item in y_true_set:
            rank = i + 1
            dcg += 1.0 / np.log2(rank + 1)

    # 计算 IDCG@k
    # 理想排名只考虑真实标签的数量，最多不超过k个
    num_true_items = len(y_true_set)
    ideal_ranks = min(num_true_items, k)
    
    idcg = 0.0
    for i in range(ideal_ranks):
        rank = i + 1
        idcg += 1.0 / np.log2(rank + 1)
        
    if idcg == 0:
        return 0.0  # 如果没有任何正确答案，或者IDCG为0，则nDCG为0
        
    return dcg / idcg


def compute_score_old(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    if isinstance(ground_truth, str):
        ground_truth = eval(ground_truth)

    # print(f"ground_truth: {ground_truth}")
    # print(f"solution_str: {solution_str}")

    solution_str = solution_str.split("</tool_response>")[-1]

    model_found_locs = extract_code_blocks(solution_str)
    model_found_locs_separated = extract_locs(model_found_locs)
    # print(f"model_found_locs_separated: {model_found_locs_separated}")
    vis_func_list = []
    hit_func_list = []
    golden_func_num = sum(len(v) for v in ground_truth.values())

    # remove func: in ground_truth
    processed_ground_truth = {}
    for file in ground_truth:
        processed_ground_truth[file] = []
        for func in ground_truth[file]:
            processed_ground_truth[file].append(func.split(": ")[-1])
    ground_truth = processed_ground_truth

    for file in model_found_locs_separated:
        for func in model_found_locs_separated[file]:
            
            # func = func.split(" ")[0] + " " + func.split(".")[-1]
            # print(f"[current] func: {func}")
            func = func.split(": ")[-1]
            if func in vis_func_list:
                continue
            vis_func_list.append(func)
            if file in ground_truth and func in ground_truth[file]:
                hit_func_list.append(func)
            if len(vis_func_list) >= 5:
                break
        if len(vis_func_list) >= 5:
            break
    if golden_func_num == 0:
        return 0.0
    # print(f"score: {float(len(hit_func_list)  /golden_func_num)}")
    return float(len(hit_func_list)  /golden_func_num)

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    if isinstance(ground_truth, str):
        ground_truth = eval(ground_truth)

    # print(f"ground_truth: {ground_truth}")
    # print(f"solution_str: {solution_str}")

    solution_str = solution_str.split("</tool_response>")[-1]

    model_found_locs = extract_code_blocks(solution_str)
    model_found_locs_separated = extract_locs(model_found_locs)
    # print(f"model_found_locs_separated: {model_found_locs_separated}")

    vis_func_list = []
    hit_func_list = []
    golden_func_num = sum(len(v) for v in ground_truth.values())

    ground_truth_set = set()
    for file in ground_truth:
        for func in ground_truth[file]:
            func = func.split(": ")[-1]
            ground_truth_func = f"{file}::{func}"
            if ground_truth_func not in ground_truth_set:
                ground_truth_set.add(ground_truth_func)

    pred_func_list = []
    for file in model_found_locs_separated:
        for func in model_found_locs_separated[file]:
            func = func.split(": ")[-1]
            pred_func_str = f"{file}::{func}"
            if pred_func_str not in pred_func_list:
                pred_func_list.append(pred_func_str)
    # print(f"score: {float(len(hit_func_list)  /golden_func_num)}")
    score = get_ndcg_reward(pred_func_list, ground_truth_set)
    return score
