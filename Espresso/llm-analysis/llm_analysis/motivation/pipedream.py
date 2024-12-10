"""
模型分割：
1. 确定GPU放置顺序
   * 暴力放置，获得全部
2. 确定model分割
   * 跑所有可能的答案。
"""
import os
import json
import sys
import itertools
from collections import defaultdict
from itertools import product
from typing import List, Dict, Any, List, Tuple
from pathlib import Path
current_dir = Path(__file__).resolve()
parent_dir = current_dir.parent.parent.parent
sys.path.append(str(parent_dir))
from config.utils import get_unique_gpus,load_model_from_config
from gpu import GPU
from GPU_resource_configuration import test_gpu_performance,get_topK_GPU_config
from plan import Plan
from model import Model
from model_partition_algorithm import profile_latency
from config.utils import load_gpus_from_config,get_unique_gpus
from constants import *
import llm_analysis.portal


def get_gpu_layer_distribution(
    best_partition: List[List[float]], 
    latency: Dict[str, List[float]], 
    gpus: List[GPU],  # 假设GPU是一个自定义类
    model_layer: int,
    parallel_status: str
) -> List[int]:
    """
    计算每个GPU的层分布。
    :param dp: 动态规划表。
    :param latency: GPU不同层的延迟。
    :param gpus: GPU列表。
    :param model_layer: 模型层数。
    :param parallel_status: 并行状态。
    :return: 每个GPU分配的层数列表。
    """
    num_gpus = len(gpus)
    remaining_layers = model_layer
    gpu_layer_distribution = [0] * num_gpus

    for i in range(num_gpus, 0, -1):
        for k in range(0, remaining_layers + 1):
            layer = remaining_layers - k
            # 因为要考虑embedding 和 最后一层的embedding，所以可以考虑这里latency直接使用llm-analysis测量
            if best_partition[i][remaining_layers] == max(best_partition[i - 1][k], latency[gpus[i - 1].name][layer]):
                gpu_layer_distribution[i - 1] = layer
                remaining_layers -= layer
                break

    return gpu_layer_distribution

def _compute_optimal_gpu_layout(
    model_layer: int, 
    latency: Dict[str, List[float]], 
    gpus: List[GPU],  # 假设GPU是一个自定义类
    parallel_status: str
) -> List[int]:
    """
    计算最优的GPU布局。
    :param model_layer: 模型层数。
    :param latency: GPU之间的延迟。
    :param gpus: GPU列表。
    :param parallel_status: 并行状态。
    :return: 每个GPU分配的层数列表。
    """
    num_gpus = len(gpus)
    best_partition = [[float('inf')] * (model_layer + 1) for _ in range(num_gpus + 1)]
    best_partition[0][0] = 0
    for i in range(1, num_gpus + 1):
        for j in range(0, model_layer + 1):
            best_partition[i][j] = best_partition[i - 1][j]
            for k in range(max(0, j - gpus[i - 1].max_layers[parallel_status]), j + 1):
                layer = j - k
                best_partition[i][j] = min(best_partition[i][j], max(best_partition[i - 1][k], latency[gpus[i - 1].name][layer]))

    return get_gpu_layer_distribution(best_partition, latency, gpus, model_layer, parallel_status)


def pipedream_partition(permutation, gpu_num, pp_size, unique_gpu_array,latency, cur_config, model):
    gpu_permutation = []
    money_cost = 0
    for x in permutation:
        for _ in range(gpu_num[unique_gpu_array[x].name]):
            gpu_permutation.append(unique_gpu_array[x])
            money_cost += unique_gpu_array[x].cost * model.dp_size * model.tp_size
    money_cost /= 3600 #gpu.cost 单位是 $/h 而 latency单位是s
    gpu_layer_distribution = _compute_optimal_gpu_layout(
        model_layer=model.model_layer,
        latency=latency,
        gpus=gpu_permutation,
        parallel_status=f"{pp_size}"
    )
    stages_config = ";".join(str(num) for num in gpu_layer_distribution)
    gpus_config = ";".join(gpu.name for gpu in gpu_permutation)
    print(stages_config,gpus_config,"gpu_permutation = ",gpu_permutation)
    flag, calculated_latency = llm_analysis.portal.train(
        model_name=model.model_name,
        partitions=stages_config,
        gpu_name=gpus_config,
        tp_size=model.tp_size,
        pp_size=pp_size,
        dp_size=model.dp_size,
        sp_size=1,
        gradient_accumulation_steps=model.gradient_accumulation_steps,
        batch_size=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
        seq_len=model.seq_len,
        activation_recomputation=2,
        output_dir="./testData"
    )
    if not flag:
        calculated_latency = float('inf')
    # 更新当前配置
    cur_config.update({
        "model_name": model.model_name,
        "partitions": stages_config,
        "gpu_name": gpus_config,
        "tp_size": model.tp_size,
        "pp_size": pp_size,
        "dp_size": model.dp_size,
        "money_cost": money_cost * calculated_latency,
        "time_cost": calculated_latency
    })
    return calculated_latency

def pipedream_find_best_plan(model,plan,slo,total_latency,perm=None):
    unique_gpu_array, gpu_num = get_unique_gpus(plan)
    pp_size = plan.pp_size
    dp_size = plan.dp_size
    tp_size = plan.tp_size
    num_points = len(unique_gpu_array)
    if not perm:
        perm = [x for x in range(num_points)]
    print(f"plan = {plan} unique_gpu_array = {unique_gpu_array} perm = {perm} ")
    min_money_cost = float("inf")
    cur_config = {'money_cost':float("inf")}
    latency_cost = pipedream_partition(
        perm, gpu_num, pp_size, unique_gpu_array, total_latency, cur_config, model
    )
    return cur_config

if __name__ == "__main__":
    gpus = load_gpus_from_config()
    model = load_model_from_config('./model_config/llama_1.3b.json')
    topK = 10
    slo = 30
    plans = get_topK_GPU_config(gpus=gpus,model=model, slo=slo, topK=topK)
    print('*'*20,f"plans {len(plans)}",'*'*20)
    total_latency = profile_latency(
        model=model,
        gpus=gpus
    )
    # 初始化答案列表
    pipedream_best_plan = {}
    min_money_cost = float("inf")
    for plan in plans:
        pipedream_plan = pipedream_find_best_plan(model,plan,slo,total_latency)
        if pipedream_plan['money_cost'] < min_money_cost:
            min_money_cost = pipedream_plan['money_cost']
            pipedream_best_plan = pipedream_plan.copy()

    print(f"pipedream最佳计划是：{pipedream_best_plan}")
    with open(os.path.join(EXPR2_PLAN_DIR, "pipedream_best_plan.json"), 'w') as f:
        json.dump(pipedream_best_plan, f, indent=4)