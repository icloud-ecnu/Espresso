from pathlib import Path
import sys
current_dir = Path(__file__).resolve()
parent_dir = current_dir.parent.parent.parent
print("parent_dir ",parent_dir)

sys.path.append(str(parent_dir))
import json,os
from typing import List
from llm_analysis.pipeline_model import cal_pipeline_time
import llm_analysis.analysis
from Espresso.utils import Plan,GPU,ResourcePlan,Model
from Espresso.utils import load_gpus_from_config,gpu_to_dict,plan_to_dict,load_model_from_config,profile_latency,test_gpu_detail

import sys
from Espresso.constants import PP_SIZES, DP_SIZES, TP_SIZES, PP_COMM, TOPK_GPU_CONFIG_FILE, BASE_DIR, inf
import heapq,copy
from Espresso.StagePlacer import StageGPUmapper



def get_transformer_layers_per_gpu(model,gpu,delta = 1.0,model_config = None):
    left, right = 1, model.model_layer
    optimal_layers = -1
    while left <= right:
        mid = (left + right) // 2
        training_data = llm_analysis.analysis.train(
            model_name=model.model_name,
            gpu_name=gpu.name,
            tp_size=model.tp_size,
            pp_size=1,
            dp_size=model.dp_size,
            gradient_accumulation_steps=model.gradient_accumulation_steps,
            batch_size_per_gpu=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
            num_layers_per_gpu=mid,
            num_microbatch=1,
            flops_efficiency=gpu.flops_efficiency,
            hbm_memory_efficiency=gpu.hbm_memory_efficiency,
            seq_len=model.seq_len,
            activation_recomputation=2,
            log_level="CRITICAL",
            first_embedding=True,
            last_embedding=False,
            model_config = model_config
        )
        total_memory = training_data['total_memory']
        if total_memory < gpu.mem_per_GPU_in_GB * (1024**3) * delta:
            left = mid + 1
            optimal_layers = mid
        else:
            right = mid - 1
    return optimal_layers    

def get_transformer_layers(model,gpus,delta,model_config = None):
    m = len(gpus)
    return sum([get_transformer_layers_per_gpu(model,gpus[i],delta,model_config) for i in range(m)])


def callMapper(gpu_config,model,slo,pp_size,beta=0.8,model_config=None):
    if gpu_config == None or len(gpu_config) == 0:
        return {"money_cost":inf,"time_cost":inf,"success":False}
    
    tmp = model.model_layer
    if beta != -1:
        model.model_layer = get_transformer_layers(model,gpu_config,beta,model_config)
    
    moneyCost = 0
    for x in gpu_config:
        moneyCost += x.cost * model.dp_size * model.tp_size

    plan = Plan(0, gpu_config, len(gpu_config), model.dp_size, model.tp_size)
    strategy,timeCost,direction = StageGPUmapper(plan,model,model_config)
    model.model_layer = tmp # 恢复layer
    layer_distribution = [x[1] for x in strategy]
    gpus_permutation = [x[0] for x in strategy]
    cur_config = {
        "model_name": model.model_name,
        "gpus_permutation": gpus_permutation,
        "tp_size": model.tp_size,
        "pp_size": pp_size,
        "dp_size": model.dp_size,
        "money_cost": moneyCost * timeCost/3600,
        "time_cost": timeCost,
        "layer_distribution":layer_distribution,
        "success":False
    }
    if timeCost < slo:
        cur_config.update({"success":True})
    return cur_config


def getModelTotalFLOPs(model):
    return 72 * model.global_batch_size * model.model_layer * model.seq_len * model.hidden_dim * model.hidden_dim * (1 + model.seq_len / (6 * model.hidden_dim)+ model.vocab_size / (12 * model.hidden_dim * model.model_layer))

def minCostEstimator(model,gpu_config,slo,pp_size,minMem,beta=0.8,model_config=None):
    if gpu_config == None or len(gpu_config) == 0: return inf,inf
    tmp = model.model_layer
    if beta != -1:
        model.model_layer = get_transformer_layers(model,gpu_config,beta,model_config)
    
    moneyCost = 0
    r = 0
    for x in gpu_config:
        moneyCost += x.cost
        r += x.peak_fp16_TFLOPS * model.dp_size
    moneyCost /= 3600
    total_flop = getModelTotalFLOPs(model) / 10**12
    Mb = model.gradient_accumulation_steps
    gp = 0.35
    p = Mb / (Mb + pp_size - 1) * gp
    Tmin = total_flop/(p * r)
    model.model_layer = tmp
    
    memCost = sum([x.mem_per_GPU_in_GB*model.dp_size for x in gpu_config])
    if memCost < minMem:
        return inf, inf

    if Tmin < slo:
        return Tmin * moneyCost, Tmin
    else: return inf, Tmin

def checkLimit(gpu_config,gpus,model):
    for i in range(len(gpus)):
        cnt = 0
        for gpu in gpu_config:
            if gpu.name == gpus[i].name:
                cnt += model.dp_size
        if cnt > limit[i]: return False
    return True
def getRapidConfig(model,gpu_config,gpus,slo,pp_size,idx,model_config):
    ans_config = {"money_cost":inf}
    for i in range(idx,len(gpus)):
        cur_gpu_config = copy.deepcopy(gpu_config)
        cur_gpu_config.extend([gpus[i] for _ in range(pp_size - len(gpu_config))]) # TODO 需要判断是否满足limit
        if not checkLimit(cur_gpu_config,gpus,model): continue
        cur_cost_config = callMapper(cur_gpu_config,model,slo,pp_size,-1,model_config)
        if cur_cost_config['success'] and ans_config["money_cost"] > cur_cost_config['money_cost']:
            ans_config = copy.deepcopy(cur_cost_config)
    return ans_config

def getMinCostRapidConfig(model,gpu_config,gpus,slo,pp_size,minMem,idx,model_config=None):
    min_cost = inf
    for i in range(idx,len(gpus)):
        cur_gpu_config = copy.deepcopy(gpu_config)
        cur_gpu_config.extend([gpus[i] for _ in range(pp_size - len(cur_gpu_config))])
        cur_cost, _ = minCostEstimator(model,cur_gpu_config,slo,pp_size,minMem,-1,model_config)
        if min_cost > cur_cost:
            min_cost = cur_cost
    return min_cost



def getKUpper(gpus,model,model_config=None):
    ans = 0
    for k in range(1,9):
        isFind = False
        for gpu in gpus:
            gpu_permutation = [gpu.name for _ in range(k)]
            gpu_layer_distribution = [model.model_layer // k for _ in range(k)]
            res_layer = model.model_layer - sum(gpu_layer_distribution)
            for idx in range(res_layer):
                gpu_layer_distribution[idx] += 1

            stages_config = ";".join(str(num) for num in gpu_layer_distribution)
            gpus_config = ";".join(x for x in gpu_permutation)
            flag, calculated_latency = llm_analysis.portal.train(
                model_name=model.model_name,
                partitions=stages_config,
                gpu_name=gpus_config,
                tp_size=model.tp_size,
                pp_size=k,
                dp_size=model.dp_size,
                sp_size=1,
                gradient_accumulation_steps=model.gradient_accumulation_steps,
                batch_size=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
                seq_len=model.seq_len,
                activation_recomputation=2,
                output_dir="./testData",
                model_config=model_config
            )
            if flag:
                isFind = True
                break
        if isFind:
            ans = max(ans,k)
    if ans == 0: ans = 8
    return ans


def getKLower(gpus,model,model_config=None):
    for k in range(1,9):
        for gpu in gpus:
            gpu_permutation = [gpu.name for _ in range(k)]
            gpu_layer_distribution = [model.model_layer // k for _ in range(k)]
            res_layer = model.model_layer - sum(gpu_layer_distribution)
            for idx in range(res_layer):
                gpu_layer_distribution[idx] += 1

            stages_config = ";".join(str(num) for num in gpu_layer_distribution)
            gpus_config = ";".join(x for x in gpu_permutation)
            flag, calculated_latency = llm_analysis.portal.train(
                model_name=model.model_name,
                partitions=stages_config,
                gpu_name=gpus_config,
                tp_size=model.tp_size,
                pp_size=k,
                dp_size=model.dp_size,
                sp_size=1,
                gradient_accumulation_steps=model.gradient_accumulation_steps,
                batch_size=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
                seq_len=model.seq_len,
                activation_recomputation=2,
                output_dir="./testData",
                model_config=model_config
            )
            if flag:
                return k
    return 9

def getModelLeastMem(model,pp_size,gpu,model_config=None):
    gpu_permutation = [gpu.name for _ in range(pp_size)]
    gpu_layer_distribution = [model.model_layer // pp_size for _ in range(pp_size)]
    res_layer = model.model_layer - sum(gpu_layer_distribution)
    for idx in range(res_layer):
        gpu_layer_distribution[idx] += 1
    stages_config = ";".join(str(num) for num in gpu_layer_distribution)
    gpus_config = ";".join(x for x in gpu_permutation)
    flag, calculated_latency,breakdownInfo = llm_analysis.portal.train(
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
        model_config=model_config,
        breakdown=True
    )
    memory = 0
    # print(breakdownInfo)
    for idx in range(pp_size):
        memory += breakdownInfo[str(idx)]['mem']
    memory /= (1024 ** 3)
    return memory

limit = [16,8,16,16,0]
def gpu_allocator(model,beta=0.8,model_config=None):
    slo = model.slo
    gpus = load_gpus_from_config()
    min_config = {"money_cost":inf}
    for edp_size in range(0,4):
        dp_size = 2**edp_size
        if model.gradient_accumulation_steps * dp_size > model.global_batch_size: continue
        model.dp_size = dp_size
        test_gpu_detail( # 出现huggingface 找不到model的问题，一般是没有传model_config
            model=model,
            gpus=gpus,
            model_config=model_config
        )
        k_lower, k_upper = getKLower(gpus,model,model_config),getKUpper(gpus,model,model_config)
        for pp_size in range(k_lower,k_upper):
            tmp_config = getRapidConfig(model,[],gpus,slo,pp_size,0,model_config)
            if tmp_config['money_cost'] < min_config['money_cost']:
                min_config = copy.deepcopy(tmp_config)
            priority_queue = [ResourcePlan(0,[],{"money_cost":inf},0,-1)]
            if test_overhead or testBP or testCP: minMem = int(1e10) #TCP
            else: minMem = getModelLeastMem(model,pp_size,gpus[0],model_config=model_config)
            while priority_queue:
                topv = heapq.heappop(priority_queue)
                gpu_config = topv.gpu_config
                stage_idx = topv.stage_idx
                idx = topv.idx
                if stage_idx == pp_size:
                    tmp_config = callMapper(gpu_config,model,slo,pp_size,-1,model_config)
                    if tmp_config['success'] and tmp_config['money_cost'] < min_config['money_cost']:
                        min_config = copy.deepcopy(tmp_config)
                    continue

                if idx >= len(gpus): continue
                if stage_idx >= pp_size: continue
                if len(gpu_config) > 0:
                    curCost, tmin = minCostEstimator(model,gpu_config,slo,pp_size,minMem,beta,model_config)
                    if curCost > min_config['money_cost']:# CP
                        continue
                    if tmin > slo: # TCP
                        continue

                for num in range(0,min(limit[idx+1]//dp_size,pp_size - len(gpu_config))+1):
                    cur_gpu_config = copy.deepcopy(gpu_config)
                    cur_gpu_config.extend([gpus[idx+1] for _ in range(num)])
                    tmp_cost = getMinCostRapidConfig(model,cur_gpu_config,gpus,slo,pp_size,minMem,idx,model_config)
                    if tmp_cost >= inf: #TCP
                        continue
                    heapq.heappush(priority_queue, ResourcePlan(tmp_cost,cur_gpu_config,{},stage_idx+num,idx+1))
    return min_config

import time

if __name__ == "__main__":
    modelname = "llama_3b"
    model = load_model_from_config(f'./model_config/{modelname}.json')
    start = time.time()
    min_cost = gpu_allocator(model)
    end = time.time()
    print(f"min_cost = {min_cost} duration = {end - start}")


        
