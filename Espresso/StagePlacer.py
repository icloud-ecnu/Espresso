import sys,copy
from typing import List, Dict, Any, List, Tuple
from pathlib import Path
current_dir = Path(__file__).resolve()
parent_dir = current_dir.parent.parent.parent
sys.path.append(str(parent_dir))
from Espresso.utils import GPU,Model,Plan
from Espresso.utils import load_model_from_config,load_gpus_from_config,test_gpu_detail, profile_latency,profile_allreduce,get_unique_gpus
import llm_analysis.portal
import heapq
from collections import defaultdict

inf = 10000000000


def initializeGPULoadTable(
    GPUs: Dict[str, GPU],
    resource_plan: Dict[str, int],
    pp_size:int,
    model: Model
):
    # print(f"resource_plan = {resource_plan}")
    # GPU type: flop,mem[1,pp_size]
    GPULoadTable = {}
    totalFlops = 0
    for x in resource_plan.keys():
        totalFlops += GPUs[x].peak_fp16_TFLOPS * resource_plan[x]
    for x in resource_plan.keys():
        if resource_plan[x] == 0: GPULoadTable[x] = [0]
        else: GPULoadTable[x] = [int((GPUs[x].peak_fp16_TFLOPS / totalFlops) * model.model_layer)] # flop
        # print(f"name = {x} layer = {int((GPUs[x].peak_fp16_TFLOPS / totalFlops) * model.model_layer)}")
        for i in range(pp_size):
            GPULoadTable[x].append(GPUs[x].max_layers[f"{pp_size}_{i}"])
    return GPULoadTable

def estimater(model,tmpStrategy,model_config=None):
    # print(f"model = {model} tmpStrategy = {tmpStrategy}")
    pp_size = len(tmpStrategy)
    gpu_permutation = [x[0] for x in tmpStrategy]
    gpu_layer_distribution = [x[1] for x in tmpStrategy]
    stages_config = ";".join(str(num) for num in gpu_layer_distribution)
    gpus_config = ";".join(gpu for gpu in gpu_permutation)
    # print(stages_config,gpus_config,"gpu_permutation = ",gpu_permutation)
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
        output_dir="./testData",
        model_config=model_config
    )
    if not flag:
        calculated_latency = inf
    return calculated_latency
    
# 输入：资源配置，GPU集合；pp size；model
def StageGPUmapper(
    plan: Plan,
    model: Model,
    model_config=None
):
    # print(plan.gpu_config)
    # import ipdb
    # ipdb.set_trace()

    GPUs = {}
    resource_plan = defaultdict(int)
    for gpu in plan.gpu_config:
        GPUs[gpu.name] = gpu
        resource_plan[gpu.name] += 1

    pp_size = plan.pp_size
    computeTime = profile_latency(
        model=model,
        gpus=GPUs.values(),
        model_config=model_config
    )
    allreduce = profile_allreduce(
        model=model,
        gpu=plan.gpu_config[0],
        model_config=model_config
    )
    if model.dp_size == 1:
        for x in allreduce.keys():
            for y in allreduce[x].keys():
                allreduce[x][y] = 0


    strategy = []
    timeCost = inf
    GPULoadTable = initializeGPULoadTable(GPUs,resource_plan,pp_size,model)
    
    # print(f"GPULoadTable = {GPULoadTable}")
    # print(f"allreduce = {allreduce}")
    direction = -1
    for nowlayer in range(1,model.model_layer+1):# 32 * 8 * 3
        gpus_permutation = []
        cur_direction = 0
        tmp_plan = copy.deepcopy(resource_plan)
        flag = True
        partition_table = [] #[max,flop] comm,mem
        res_layer = model.model_layer
        comm_time = 0
        for stage_idx in range(pp_size):
            GPUList = []
            if stage_idx == 0:
                for x in tmp_plan:
                    if tmp_plan[x] > 0 and GPULoadTable[x][1] >= nowlayer:
                        GPUList.append([GPULoadTable[x][0]-nowlayer,nowlayer,GPULoadTable[x][1],x]) 
            else:
                comm_layer = 0
                left, right = 1,res_layer
                while left <= right:# 可以找comm layer
                    mid = (left+right)//2
                    if allreduce['mid'][mid] < comm_time:
                        comm_layer = mid
                        left = mid + 1
                    else:
                        right = mid - 1

                for x in tmp_plan:
                    if tmp_plan[x] > 0:
                        valid_layer = min(comm_layer,GPULoadTable[x][stage_idx + 1])
                        GPUList.append([GPULoadTable[x][0]-valid_layer,comm_layer,GPULoadTable[x][stage_idx + 1],x]) # sortedkey,layer,GPUname
            # GPUList
            GPUList = sorted(GPUList,key=lambda x: x[0])
            if len(GPUList) == 0: 
                flag = False
                break
            _,cur_comm_layer,cur_mem_layer,GPUname = GPUList[0]
            tmp_plan[GPUname] -= 1
            gpus_permutation.append(GPUname)
            partition_table.append([cur_comm_layer,cur_mem_layer,GPULoadTable[GPUname][0]]) # comm,mem,tflop
            min_layer = min(cur_comm_layer,cur_mem_layer,GPULoadTable[GPUname][0])
            comm_time += computeTime[GPUname][min_layer] * 2.7 # backward time
            res_layer -= min_layer
            if stage_idx == 0:
                comm_time += allreduce['first'][min_layer]

        
        if not flag:
            break
        res_layer = model.model_layer
        layer_distribution = [min(x) for x in partition_table]
        layer_distribution[0] = partition_table[0][0]
        res_layer -= sum(layer_distribution)
        
        
        priority_queue = []
        for idx,x in enumerate(partition_table):
            heapq.heappush(priority_queue, (layer_distribution[idx] - min(x[0],x[1]),layer_distribution[idx],idx))
        while priority_queue and res_layer > 0:
            priority,layer,idx = heapq.heappop(priority_queue)
            layer_distribution[idx] += 1
            heapq.heappush(priority_queue,(priority+1,layer+1,idx))
            res_layer -= 1
        
        tmpStrategy = [[gpus_permutation[x],layer_distribution[x]] for x in range(pp_size)]
        curTimeCost = estimater(model,tmpStrategy,model_config=model_config)
        if curTimeCost < timeCost:
            timeCost = curTimeCost
            strategy = copy.deepcopy(tmpStrategy)
            direction=cur_direction
    return strategy,timeCost,direction >=0 # > 0 mem, = 0 , <0 flop


if __name__ == "__main__":
    gpus = load_gpus_from_config()
    modelname = "llama_1.3b"
    model = load_model_from_config(f'./model_config/{modelname}.json')
    model.dp_size = 2
    print(f"mainmodel = {model}")
    # 2 3 0 1
    gpu_config = [gpus[0],gpus[2],gpus[3]]
    test_gpu_detail(
        model=model,
        gpus=gpus,
    )
    computeTime = profile_latency(
        model=model,
        gpus=gpus,
    )
    # import ipdb 
    # ipdb.set_trace()
    plan = Plan(0, gpu_config, len(gpu_config), model.dp_size, model.tp_size)
    BestConfig = StageGPUmapper(plan,model)
    
    print(f"ans = {BestConfig}") # test
    ans = []
    moneyCost=0
    mp = {"3090-pcie4-24gb":2.5,'a6000-pcie4-48gb':6,"a4000-pcie4-16gb":2,"a30-pcie-24gb":4}
    for x in BestConfig[0]:
        x = x[0]
        moneyCost += mp[x]
        x = x.split('-')[0]
        ans.append(x)
    print("-".join(ans))
    print(BestConfig[1])
    print(f"{model.dp_size} * {moneyCost}")




"""
ans = {'model_name': 'llama_3b', 'gpus_permutation': ['a6000-pcie4-48gb', 'a6000-pcie4-48gb'], 'tp_size': 1, 'pp_size': 2, 'dp_size': 4, 'money_cost': 0.44850177708828026, 'time_cost': 67.27526656324204, 'layer_distribution': [19, 19], 'success': True} duration = 335.45323729515076
160 * 8 = 1200
170
(48 * 3 + 24 ) * 2 = 340
def StageGPUmapper(
    GPUs: Dict[str, GPU],
    resource_plan: Dict[str, int],
    pp_size:int,
    model: Model,
    allreduce: List[float],
    computeTime: Dict,
    model_config=None
):
"""