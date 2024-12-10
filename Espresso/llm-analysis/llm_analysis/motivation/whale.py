"""
模型分割：
1. 确定GPU放置顺序
   * 暴力放置，获得全部
2. 确定model分割
   * 跑所有可能的答案。
"""
import sys
import os
import json
import itertools
from llm_analysis.pipeline_model import cal_op
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

def whale_partition(permutation, gpu_num, pp_size, unique_gpu_array,latency, cur_config, model):
    gpu_permutation = []
    money_cost = 0
    for x in permutation:
        for _ in range(gpu_num[unique_gpu_array[x].name]):
            gpu_permutation.append(unique_gpu_array[x])
            money_cost += unique_gpu_array[x].cost * model.dp_size * model.tp_size
    money_cost /= 3600 #gpu.cost 单位是 $/h 而 latency单位是s\
    gpu_permutation.sort(key=lambda gpu: gpu.mem_per_GPU_in_GB, reverse=True)
    device_flop = [gpu.peak_fp16_TFLOPS for gpu in gpu_permutation]
    device_memory = [gpu.mem_per_GPU_in_GB * (1024**3) for gpu in gpu_permutation]
    data = llm_analysis.analysis.train(
        model_name=model.model_name, gpu_name=gpu_permutation[0].name, tp_size=model.tp_size, 
        pp_size=1, dp_size=model.dp_size, 
        gradient_accumulation_steps=model.gradient_accumulation_steps,
        batch_size_per_gpu=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
        num_layers_per_gpu=model.model_layer, num_microbatch=0,
        flops_efficiency=gpu_permutation[0].flops_efficiency, 
        hbm_memory_efficiency=gpu_permutation[0].hbm_memory_efficiency,
        seq_len=model.seq_len, activation_recomputation=2, log_level="CRITICAL",
        first_embedding=True,
        last_embedding=True,
    )
    TGmem = data['total_memory']
    TGflop = data['num_flops_total_per_micro_batch']

    load_ratios = [0] * pp_size
    distribution = [0] * pp_size
    mem_utils = [0] * pp_size
    oom_devices = []
    free_devices = []
    for i in range(pp_size):
        load_ratios[i] = device_flop[i] / sum(device_flop)
        distribution[i] = int(load_ratios[i] * model.model_layer)
        sequence = cal_op(model.gradient_accumulation_steps,pp_size,i)
        num_microbatch = 0
        for x in sequence:
            if x == 'F': num_microbatch += 1
            else: break
        training_data = llm_analysis.analysis.train(
            model_name=model.model_name, gpu_name=gpu_permutation[i].name, tp_size=model.tp_size, 
            pp_size=1, dp_size=model.dp_size, 
            gradient_accumulation_steps=model.gradient_accumulation_steps,
            batch_size_per_gpu=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
            num_layers_per_gpu=distribution[i], num_microbatch=num_microbatch,
            flops_efficiency=gpu_permutation[i].flops_efficiency, 
            hbm_memory_efficiency=gpu_permutation[i].hbm_memory_efficiency,
            seq_len=model.seq_len, activation_recomputation=2, log_level="CRITICAL",
            first_embedding= True if i == 0 else False,
            last_embedding=True if i == pp_size - 1 else False,
        )
        mem_utils[i] = training_data['total_memory']
        if training_data['memory_left'] < 0:
            oom_devices.append(i)
        else:
            free_devices.append(i)
    res = model.model_layer - sum(distribution)
    idx = pp_size - 1
    while res != 0 and idx >= 0:
        if idx not in oom_devices:
            res -= 1
            distribution[idx] += 1
            training_data = llm_analysis.analysis.train(
                model_name=model.model_name, gpu_name=gpu_permutation[idx].name, tp_size=model.tp_size, 
                pp_size=1, dp_size=model.dp_size, 
                gradient_accumulation_steps=model.gradient_accumulation_steps,
                batch_size_per_gpu=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
                num_layers_per_gpu=distribution[idx], num_microbatch=num_microbatch,
                flops_efficiency=gpu_permutation[idx].flops_efficiency, 
                hbm_memory_efficiency=gpu_permutation[idx].hbm_memory_efficiency,
                seq_len=model.seq_len, activation_recomputation=2, log_level="CRITICAL",
                first_embedding= True if idx == 0 else False,
                last_embedding=True if idx == pp_size - 1 else False,
            )
            mem_utils[idx] = training_data['total_memory']
            if training_data['memory_left'] < 0:
                oom_devices.append(idx)
        else:
            idx -= 1

    print(distribution,oom_devices,free_devices)
    while oom_devices and free_devices:
        peak_device = max(oom_devices, key=lambda x: mem_utils[x])
        valley_device = min(free_devices, key=lambda x: (distribution[x], mem_utils[x]))
        training_data_peak_device = llm_analysis.analysis.train(
            model_name=model.model_name, gpu_name=gpu_permutation[peak_device].name, tp_size=model.tp_size, 
            pp_size=1, dp_size=model.dp_size, 
            gradient_accumulation_steps=model.gradient_accumulation_steps,
            batch_size_per_gpu=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
            num_layers_per_gpu=distribution[peak_device] - 1, num_microbatch=num_microbatch,
            flops_efficiency=gpu_permutation[peak_device].flops_efficiency, 
            hbm_memory_efficiency=gpu_permutation[peak_device].hbm_memory_efficiency,
            seq_len=model.seq_len, activation_recomputation=2, log_level="CRITICAL",
            first_embedding= True if peak_device == 0 else False,
            last_embedding=True if peak_device == pp_size - 1 else False,
        )
        training_data_valley_device = llm_analysis.analysis.train(
            model_name=model.model_name, gpu_name=gpu_permutation[valley_device].name, tp_size=model.tp_size, 
            pp_size=1, dp_size=model.dp_size, 
            gradient_accumulation_steps=model.gradient_accumulation_steps,
            batch_size_per_gpu=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
            num_layers_per_gpu=distribution[valley_device] + 1, num_microbatch=num_microbatch,
            flops_efficiency=gpu_permutation[valley_device].flops_efficiency, 
            hbm_memory_efficiency=gpu_permutation[valley_device].hbm_memory_efficiency,
            seq_len=model.seq_len, activation_recomputation=2, log_level="CRITICAL",
            first_embedding= True if valley_device == 0 else False,
            last_embedding=True if valley_device == pp_size - 1 else False,
        )
        if training_data_valley_device['memory_left'] > 0:
            distribution[valley_device] += 1
            distribution[peak_device] -= 1
            if training_data_peak_device['memory_left'] > 0:
                oom_devices.remove(peak_device)
        else:
            free_devices.remove(valley_device)
        
    print(distribution)
    stages_config = ";".join(str(num) for num in distribution)
    gpus_config = ";".join(gpu.name for gpu in gpu_permutation)
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
    # 返回计算得到的总延迟
    return calculated_latency

def whale_find_best_plan(model,plan,slo,total_latency):
    unique_gpu_array, gpu_num = get_unique_gpus(plan)
    pp_size = plan.pp_size
    dp_size = plan.dp_size
    tp_size = plan.tp_size
    num_points = len(unique_gpu_array)
    # 随便给一个顺序，比如：按照显存、flop、random之类的
    perm = [x for x in range(num_points)]
    cur_config = {'money_cost':float("inf")}
    latency_cost = whale_partition(
        perm, gpu_num, pp_size, unique_gpu_array, total_latency, cur_config, model
    )
    return cur_config

if __name__ == "__main__":
    gpus = load_gpus_from_config()
    model = load_model_from_config('./model_config/llama_1.3b_large.json')
    topK = 10
    slo = 30
    plans = get_topK_GPU_config(gpus=gpus,model=model, slo=slo, topK=topK)
    total_latency = profile_latency(
        model=model,
        gpus=gpus
    )
    # 初始化答案列表
    whale_best_plan = {}
    min_money_cost = float("inf")
    for plan in plans:
        whale_plan = whale_find_best_plan(model,plan,slo,total_latency)
        if whale_plan['money_cost'] < min_money_cost:
            min_money_cost = whale_plan['money_cost']
            whale_best_plan = whale_plan.copy()


    print(f"whale最佳计划是：{whale_best_plan}")
    with open(os.path.join(EXPR2_PLAN_DIR, "whale_best_plan.json"), 'w') as f:
        json.dump(whale_best_plan, f, indent=4)