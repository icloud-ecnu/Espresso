import sys,os,json,time
import itertools
from collections import defaultdict
from itertools import product
from typing import List, Dict, Any, List, Tuple
from pathlib import Path
current_dir = Path(__file__).resolve()
parent_dir = current_dir.parent.parent.parent
sys.path.append(str(parent_dir))
from itertools import product
import llm_analysis.portal
from llm_analysis.pipeline_model import cal_pipeline_time
from config.utils import load_gpus_from_config,get_unique_gpus,load_model_from_config
import numpy as np


def cal_latency(distribution):
    m = len(distribution)
    stages_config = ";".join(str(num) for num in distribution)
    gpus_config = ";".join(gpu_perm[x] for x in range(m))
    flag, calculated_latency,breakdownInfo = llm_analysis.portal.train(
        model_name=model_name,
        partitions=stages_config,
        gpu_name=gpus_config,
        tp_size=1,
        pp_size=m,
        dp_size=1,
        sp_size=1,
        gradient_accumulation_steps=4,
        batch_size=16,
        seq_len=1800,
        activation_recomputation=2,
        output_dir="./testData",
        breakdown=True
    )
    return flag,calculated_latency,breakdownInfo


def test_cal_pipeline_time(model_layer, gpu_perm):
    # 获取GPU数量
    num_gpus = len(gpu_perm)

    # 生成所有可能的层分配方案
    possible_distributions = []
    for distribution in product(range(model_layer + 1), repeat=num_gpus):
        if sum(distribution) == model_layer:
            possible_distributions.append(distribution)

    # 遍历所有方案,调用train函数预测时间开销
    cost = float('inf')
    result = []
    info = ""
    for distribution in possible_distributions:
        flag,calculated_latency,breakdownInfo = cal_latency(distribution)
        if flag and calculated_latency < cost:
            cost = calculated_latency
            result = distribution
            info = breakdownInfo


    print(f'result = {result} cost={cost} info = {info}')
    ans = []
    for x in range(num_gpus):
        ans.append(result[str(x)]['latency'])
    print(ans)



# model = load_model_from_config('./model_config/sheared_llama_2.7b.json')
# model = load_model_from_config('./model_config/open_llama_3b_v2.json')
model = load_model_from_config('./model_config/open_llama_3b_v2.json')
# gpu_perm = ['a30-pcie-24gb','a30-pcie-24gb',"a4000-pcie4-16gb"]
# gpu_perm = ['a30-pcie-24gb','a30-pcie-24gb','a30-pcie-24gb']
# gpu_perm = ["a4000-pcie4-16gb","a4000-pcie4-16gb","a4000-pcie4-16gb"]
# gpu_perm = ["a6000-pcie4-48gb",'a6000-pcie4-48gb','a6000-pcie4-48gb','a6000-pcie4-48gb']
# distributions = [[12,11,9],[12,9,11],[9,11,12],[9,12,11],[11,12,9],[11,9,12]]
gpu_perm = ["a6000-pcie4-48gb",'a30-pcie-24gb','a30-pcie-24gb']
gpu_perm = ['a30-pcie-24gb',"a4000-pcie4-16gb","a6000-pcie4-48gb"]
gpu_perm = ["a6000-pcie4-48gb","a4000-pcie4-16gb",'a30-pcie-24gb']
gpu_perm = ["a6000-pcie4-48gb",'a30-pcie-24gb','a30-pcie-24gb',"a4000-pcie4-16gb"]
gpu_perm = ["a6000-pcie4-48gb","a4000-pcie4-16gb",'a30-pcie-24gb','a30-pcie-24gb']
# gpu_perm = ['a30-pcie-24gb','a30-pcie-24gb',"a4000-pcie4-16gb","a4000-pcie4-16gb"]
gpu_perm = ["a4000-pcie4-16gb","a4000-pcie4-16gb","a4000-pcie4-16gb","a4000-pcie4-16gb"]
distribution = [12,12,9,8,7]
distribution = [8,9,9,6]
# gpu_perm = ['a30-pcie-24gb','a30-pcie-24gb','a30-pcie-24gb']
# distribution = [11,11,10]

gpu_perm = ["a6000-pcie4-48gb","a6000-pcie4-48gb"]
distribution = [17,15]


gpu_perm = ["a6000-pcie4-48gb",'a30-pcie-24gb','a30-pcie-24gb']
distribution = [20,9,3]

# gpu_perm = ["a6000-pcie4-48gb",'a30-pcie-24gb']
# distribution = [21,11]

# gpu_perm = ['a30-pcie-24gb','a30-pcie-24gb',"a4000-pcie4-16gb"]
# distribution = [14,14,4]

# gpu_perm = ["a6000-pcie4-48gb"]
# distribution = [32]
# distribution = [12,11,5,4]
# distribution = [10,10,6,6]


gpu_perm = ["a6000-pcie4-48gb","a6000-pcie4-48gb","a4000-pcie4-16gb"]
distribution = [12,12,1]

pp_size = len(distribution)
stages_config = ";".join(str(num) for num in distribution)
gpus_config = ";".join(gpu_perm[x] for x in range(pp_size))
flag, calculated_latency, breakdownInfo = llm_analysis.portal.train(
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
    output_dir="./testData",
    breakdown=True,
    model_layer=25
)
print(f"flag = {flag}")
print(f"breakdownInfo = {breakdownInfo}")
print(f"calculated_latency = {calculated_latency}")

time_data = []
for key in breakdownInfo:
    time_data.append(breakdownInfo[key]['latency'])
array = np.array(time_data)  # 将 a, b, c, d 替换为实际的值
std_deviation = np.std(array)
print(f"fwd = {time_data}")
bwd = [x*3.07 for x in time_data]
print(f"bwd = {bwd}")
print(f"std_deviation = {std_deviation}")

