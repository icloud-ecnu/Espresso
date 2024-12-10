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
from model_partition_algorithm import profile_latency,isomer_find_best_plan
from constants import *
import llm_analysis.portal
import numpy as np
import random
from optimal import force_find_best_plan
from pipedream import pipedream_find_best_plan
from whale import whale_find_best_plan

model = load_model_from_config('./model_config/open_llama_3b_v2.json')
slo = 20
gpus = load_gpus_from_config()
test_gpu_performance(
    model=model,
    slo=slo,
    gpus=gpus,
)
gpu_config = [gpus[1],gpus[1],gpus[2]]
# gpu_config = [gpus[1],gpus[0],gpus[0],gpus[1],gpus[0]]
num_points = 3
plans = [Plan(0, gpu_config, len(gpu_config), model.dp_size, model.tp_size)]
print(gpu_config)
slo = 40
total_latency = profile_latency(
    model=model,
    gpus=gpus
)
for i, plan in enumerate(plans):
    # isomer_best_plan = isomer_find_best_plan(model,plan,total_latency)
    # print(f"isomer_best_plan = {isomer_best_plan}")

    whale_best_plan = whale_find_best_plan(model,plan,slo,total_latency)
    isomer_best_plan = isomer_find_best_plan(model,plan,total_latency)

    all_permutations = list(itertools.permutations(range(num_points)))
    pipedream_best_plan = {'time_cost':0}
    for perm in all_permutations:
        pipedream_plan = pipedream_find_best_plan(model,plan,slo,total_latency,perm=perm)
        if pipedream_plan['time_cost'] < slo and pipedream_plan['time_cost'] > pipedream_best_plan['time_cost']:
            pipedream_best_plan.clear()
            pipedream_best_plan.update(pipedream_plan)
    
    optimal_best_plan = force_find_best_plan(model,plan,slo)

    print(f"optimal_best_plan = {optimal_best_plan}")
    print(f"isomer_best_plan = {isomer_best_plan}")
    print(f"pipedream_best_plan = {pipedream_best_plan}")
    print(f"whale_best_plan = {whale_best_plan}")




