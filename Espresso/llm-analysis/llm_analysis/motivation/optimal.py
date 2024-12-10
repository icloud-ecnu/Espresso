"""
模型分割：
1. 确定GPU放置顺序
   * 暴力放置，获得全部
2. 确定model分割
   * 跑所有可能的答案。
"""
import sys,os,json,time
import itertools
from itertools import product
from collections import defaultdict
from typing import List, Dict, Any, List, Tuple
from pathlib import Path
current_dir = Path(__file__).resolve()
parent_dir = current_dir.parent.parent.parent
sys.path.append(str(parent_dir))
from config.utils import get_unique_gpus,load_model_from_config,load_gpus_from_config
from gpu import GPU
from GPU_resource_configuration import test_gpu_performance,get_topK_GPU_config
from plan import Plan
from model import Model
from constants import *
import llm_analysis.portal
import numpy as np
import random
from functools import lru_cache
def cal_latency(distribution,model,gpus_config,pp_size):
    stages_config = ";".join(str(num) for num in distribution)
    return _cal_latency(stages_config,model,gpus_config,pp_size)

@lru_cache(100000)
def _cal_latency(stages_config,model,gpus_config,pp_size):
    # print(distribution)
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
    return calculated_latency

def generate_list(gpu_num, total_sum):
    nums = [0] * gpu_num
    for _ in range(total_sum):
        nums[random.randint(0, gpu_num-1)] += 1
    return nums

def sa(gpu_permutation,pp_size,model):
    import math
    gpus_config = ";".join(gpu.name for gpu in gpu_permutation)
    layers_num = model.model_layer
    gpu_num = len(gpu_permutation)
    best = None
    best_ans = 1e9
    for _ in range(20):
        arrange = generate_list(gpu_num,layers_num)
        pre = cal_latency(arrange,model,gpus_config,pp_size)
        down = 0.97
        teampture = 10
        while teampture>1e-14:
            teampture = teampture*down
            idx = random.randint(0,gpu_num-1)
            new_arrange = arrange[:]
            if idx==0:
                if new_arrange[idx]>0:
                    new_arrange[idx]-=1
                    new_arrange[idx+1]+=1
            elif idx==gpu_num-1:
                if new_arrange[idx]>0:
                    new_arrange[idx]-=1
                    new_arrange[idx-1]+=1
            else:
                if new_arrange[idx]>0:
                    if random.randint(0,1)==0:
                        new_arrange[idx]-=1
                        new_arrange[idx-1]+=1
                    else:
                        new_arrange[idx]-=1
                        new_arrange[idx+1]+=1
            now = cal_latency(new_arrange,model,gpus_config,pp_size)
            e=math.exp(min(100,abs(pre-now)/teampture))
            if now<=pre:
                if best_ans>now:
                    best_ans = now
                    best = new_arrange
                # best = now
                arrange = new_arrange
                pre = now
            elif 1.0/(1+e)>random.random():
                arrange = new_arrange
                pre = now
    return best,best_ans

def force_partition(permutation, gpu_num, pp_size, unique_gpu_array, cur_config, model, slo):
    gpu_permutation = []
    money_cost = 0
    for x in permutation:
        for _ in range(gpu_num[unique_gpu_array[x].name]):
            gpu_permutation.append(unique_gpu_array[x])
            money_cost += unique_gpu_array[x].cost * model.dp_size * model.tp_size
    money_cost /= 3600 #gpu.cost 单位是 $/h 而 latency单位是s
    num_items = len(gpu_permutation)
    best_distribution, calculated_latency = sa(gpu_permutation,pp_size,model)
    print(best_distribution)
    stages_config = ";".join(str(num) for num in best_distribution)
    gpus_config = ";".join(gpu.name for gpu in gpu_permutation)
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

def force_find_best_plan(model,plan,slo):
    plan_min_money_cost=float('inf')
    unique_gpu_array, gpu_num = get_unique_gpus(plan)
    pp_size = plan.pp_size
    dp_size = plan.dp_size
    tp_size = plan.tp_size
    num_points = len(unique_gpu_array)
    all_permutations = list(itertools.permutations(range(num_points)))
    plan_force_best_plan = {}
    for perm in all_permutations:
        cur_config = {'money_cost':float("inf")}
        latency_cost = force_partition(
            perm, gpu_num, pp_size, unique_gpu_array, cur_config, model, slo
        )
        if cur_config['money_cost'] < plan_min_money_cost:
            plan_min_money_cost = cur_config['money_cost']
            plan_force_best_plan.update(cur_config)
    return plan_force_best_plan

if __name__ == "__main__":
    start = time.time()
    gpus = load_gpus_from_config()
    # model = load_model_from_config('./model_config/open_llama_3b_v2.json')
    # model = load_model_from_config('./model_config/sheared_llama_2.7b.json')
    model = load_model_from_config('./model_config/llama_1.3b_middle.json')
    topK = 10
    slo = 50
    plans = get_topK_GPU_config(gpus=gpus,model=model, slo=slo, topK=topK)
    # 初始化答案列表
    force_best_plan = {}
    min_money_cost = float("inf")
    for i, plan in enumerate(plans):
        progress_percentage = ((i + 1) / len(plans)) * 100
        force_plan = force_find_best_plan(model,plan,slo)
        if force_plan['money_cost'] < min_money_cost:
            min_money_cost = force_plan['money_cost']
            force_best_plan = force_plan.copy()
        print(f"Processing plan {i+1}/{len(plans)} ({progress_percentage:.2f}%) complete")
        print(force_plan)
    
    end = time.time()
    print(f"optimal最佳计划是：{force_best_plan}  时间花费是：{end - start}")
    force_best_plan.update({'algorithm_overhead': end - start})
    with open(os.path.join(EXPR2_PLAN_DIR, "force_best_plan.json"), 'w') as f:
        json.dump(force_best_plan, f, indent=4)