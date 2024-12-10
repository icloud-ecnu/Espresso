from Espresso.constants import *
from pathlib import Path
import json
from collections import defaultdict
from typing import List, Dict, Any, List, Tuple
import llm_analysis.portal
from llm_analysis.pipeline_model import cal_pipeline_time,cal_op


class Model:
    def __init__(self, model_name, gradient_accumulation_steps, seq_len, slo, model_layer, global_batch_size, dp_size, tp_size, num_layers, n_head, hidden_dim, vocab_size, max_seq_len, sp_size = 1, alpha = 3.0):
        self.model_name = model_name
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seq_len = seq_len
        self.slo = slo
        self.model_layer = model_layer
        self.global_batch_size = global_batch_size
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.num_layers = num_layers
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.sp_size = sp_size
        self.alpha = alpha

    
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
    
    def __str__(self):
        return (f"Model Name: {self.model_name}, Gradient Accumulation Steps: {self.gradient_accumulation_steps}, "
                f"Sequence Length: {self.seq_len}, Model Layer: {self.model_layer}, "
                f"Global Batch Size: {self.global_batch_size}, DP Size: {self.dp_size}, TP Size: {self.tp_size}, "
                f"Num Layers: {self.num_layers}, Head Number: {self.n_head}, Hidden Dimension: {self.hidden_dim}, "
                f"Vocab Size: {self.vocab_size}, Max Sequence Length: {self.max_seq_len}")


# 定义一个用于保存方案细节的数据结构
class Plan:
    def __init__(self, cost, gpu_config, pp_size, dp_size, tp_size, latency = 0, layer = 0, min_layer = 0):
        self.cost = cost                  # 总成本
        self.gpu_config = gpu_config      # 每种GPU的使用个数
        self.pp_size = pp_size            # PP大小
        self.dp_size = dp_size            # DP大小
        self.tp_size = tp_size            # TP大小
        self.latency = latency
        self.layer = layer
        self.min_layer = min_layer

    def __str__(self):
        return (f"Plan(cost: {self.cost}, GPU config: {self.gpu_config}, "
                f"PP size: {self.pp_size}, DP size: {self.dp_size}, layer: {self.layer}, min_layer: {self.min_layer})")
    
    def __repr__(self):
        return (f"Plan(cost: {self.cost}, GPU config: {self.gpu_config}, "
                f"PP size: {self.pp_size}, DP size: {self.dp_size}, TP size: {self.tp_size}, latency: {self.latency}, layer: {self.layer}, min_layer: {self.min_layer})")
    
    def __lt__(self, other):
        return self.layer * other.layer < other.layer * self.cost  


from collections import defaultdict
class GPU:
    def __init__(self, name, cost, flops_efficiency, hbm_memory_efficiency, mem_per_GPU_in_GB,peak_fp16_TFLOPS):
        self.name = name
        self.max_layers = defaultdict(int)  # 初始化为 defaultdict
        self.cost = cost
        self.total_cost = defaultdict(int)  # 修改变量名
        self.flops_efficiency = flops_efficiency
        self.hbm_memory_efficiency = hbm_memory_efficiency
        self.mem_per_GPU_in_GB = mem_per_GPU_in_GB
        self.peak_fp16_TFLOPS = peak_fp16_TFLOPS
    
    def __str__(self):
        return (f"GPU(Name: {self.name}, Cost: {self.cost}, Total Cost: {self.total_cost}, "
                f"FLOPS Efficiency: {self.flops_efficiency}, HBM Memory Efficiency: {self.hbm_memory_efficiency}, "
                f"Max Layers: {dict(self.max_layers)})")
    
    def __repr__(self):
        return self.name

# 定义一个用于保存方案细节的数据结构
class ResourcePlan:
    def __init__(self, cost, gpu_config, cost_config, stage_idx, idx):
        self.cost = cost                 
        self.gpu_config = gpu_config     
        self.cost_config = cost_config   
        self.stage_idx = stage_idx       
        self.idx = idx            

    def __lt__(self, other):
        return self.cost < other.cost

def _calculate_num_microbatch(sequence):
    num_microbatch = 0
    for char in sequence:
        if char == 'F':
            num_microbatch += 1
        else:
            break  # 当遇到非 'F' 的字符时，停止计数
    return num_microbatch

def load_gpus_from_config():
    with open(f"{BASE_DIR}/{GPU_CONFIG_FILE}", 'r') as file:
        data = json.load(file)
        gpus = []
        for name, attributes in data.items():
            gpu = GPU(name,  
                    cost=attributes['cost'],      # 暂时假设成本为100
                    flops_efficiency=attributes['flops_efficiency'],
                    hbm_memory_efficiency=attributes['hbm_memory_efficiency'],
                    mem_per_GPU_in_GB=attributes['mem_per_GPU_in_GB'],
                    peak_fp16_TFLOPS=attributes['peak_fp16_TFLOPS']
                    )
            gpus.append(gpu)
        return gpus

def load_model_from_config(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
        # print(data)
        # print(filepath)
        model = Model(
            model_name=data['model_name'],
            gradient_accumulation_steps=data['gradient_accumulation_steps'],
            seq_len=data['seq_len'],
            model_layer=data['model_layer'],
            global_batch_size=data['global_batch_size'],
            dp_size=data['dp_size'],
            tp_size=data['tp_size'],
            num_layers=data['num_layers'],
            n_head=data['n_head'],
            hidden_dim=data['hidden_dim'],
            vocab_size=data['vocab_size'],
            max_seq_len=data['max_seq_len']
        )
        return model 

def gpu_to_dict(gpu: GPU) -> dict:
    return {
        "name": gpu.name,
        "cost": gpu.cost,
        "total_cost": gpu.total_cost,
        "flops_efficiency": gpu.flops_efficiency,
        "hbm_memory_efficiency": gpu.hbm_memory_efficiency,
        "max_layers": dict(gpu.max_layers)
    }

def plan_to_dict(plan: Plan) -> dict:
    return {
        "cost": plan.cost,
        "gpu_config": [gpu_to_dict(gpu) for gpu in plan.gpu_config],
        "pp_size": plan.pp_size,
        "dp_size": plan.dp_size,
        "tp_size": plan.tp_size
    }

def get_unique_gpus(plan) -> Tuple[List[GPU], Dict[str, int]]:
    """
    获取计划中独特的GPU集合。
    :param plan: 包含GPU配置的计划。
    :return: 独特的GPU列表和每种GPU的数量。
    """
    unique_gpus = defaultdict(int)
    result = []
    for gpu in plan.gpu_config:
        if unique_gpus[gpu.name] == 0:
            result.append(gpu)
        unique_gpus[gpu.name] += 1
    return result, unique_gpus

def profile_allreduce(
    model:Model,
    gpu: GPU,
    model_config:dict=None
):  
    networkBandwidth = 2.5 * (1024 ** 3) # 需要和llm.analysis.portal里面一致
    allreduce = {}
    for idx in range(0,3):
        # idx = 0, first stage
        # idx = 1, mid
        # idx = 2, last stage
        name = "mid"
        if idx == 0: name = "first"
        if idx == 2: name = "last"
        allreduce[name] = {}
        for layer in range(0,model.model_layer+1):
            information = llm_analysis.analysis.train(
                model_name=model.model_name,
                gpu_name=gpu.name,
                tp_size=model.tp_size,
                pp_size=1,
                dp_size=model.dp_size,
                gradient_accumulation_steps=model.gradient_accumulation_steps,
                batch_size_per_gpu=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
                num_layers_per_gpu=layer,
                num_microbatch=0,
                flops_efficiency=gpu.flops_efficiency,
                hbm_memory_efficiency=gpu.hbm_memory_efficiency,
                seq_len=model.seq_len,
                activation_recomputation=2,
                log_level="CRITICAL",
                first_embedding=True if idx == 0 else False,
                last_embedding=True if idx == 2 else False,
                model_config=model_config
            )
            comm_size = information['weight_memory_layer'] * 8 * layer
            # print(comm_size)
            if idx == 0 or idx == 2:
                comm_size += information['weight_memory_embedding'] * 8
                # print(information['weight_memory_embedding'] * 8)
            allreduce[name][layer] = comm_size / networkBandwidth
    return allreduce


def profile_latency(
    model: Model, 
    gpus: List[GPU], 
    model_config:dict=None
) -> Dict[str, Dict[str, List[float]]]:
    """
    对给定模型和GPU配置，测量延迟。
    :param model: 模型。
    :param gpus: GPU列表。
    :return: 不同配置下每个GPU的延迟。
    """
    result = defaultdict(dict)
    for gpu in gpus:
        result[gpu.name][0] = 0
        for layer in range(1, model.model_layer + 1):
            data = llm_analysis.analysis.train(
                model_name=model.model_name, gpu_name=gpu.name, tp_size=model.tp_size, 
                pp_size=1, dp_size=model.dp_size, 
                gradient_accumulation_steps=model.gradient_accumulation_steps,
                batch_size_per_gpu=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
                num_layers_per_gpu=layer, num_microbatch=0,
                flops_efficiency=gpu.flops_efficiency, 
                hbm_memory_efficiency=gpu.hbm_memory_efficiency,
                seq_len=model.seq_len, activation_recomputation=2, log_level="CRITICAL",
                model_config=model_config
            )
            # print(gpu.name,data['total_memory']/1024/1024/1024,data['total_latency'],layer)
            result[gpu.name][layer] = data['total_latency']
    return result


def test_gpu_detail(
    model: Model, 
    gpus: List[GPU], 
    model_config:dict=None
) -> None:
    for gpu in gpus:
        for pp_size in PP_SIZES:
            op = [cal_op(model.gradient_accumulation_steps, pp_size, i) for i in range(pp_size)]
            num_microbatch = [_calculate_num_microbatch(op[i]) for i in range(pp_size)]
            for stage in range(pp_size):
                optimal_layers = -1
                for layer in range(1,model.model_layer+1):
                    # import ipdb
                    # ipdb.set_trace()
                    training_data = llm_analysis.analysis.train(
                        model_name=model.model_name,
                        gpu_name=gpu.name,
                        tp_size=model.tp_size,
                        pp_size=1,
                        dp_size=model.dp_size,
                        gradient_accumulation_steps=model.gradient_accumulation_steps,
                        batch_size_per_gpu=model.global_batch_size // (model.dp_size * model.gradient_accumulation_steps),
                        num_layers_per_gpu=layer,
                        num_microbatch=num_microbatch[stage],
                        flops_efficiency=gpu.flops_efficiency,
                        hbm_memory_efficiency=gpu.hbm_memory_efficiency,
                        seq_len=model.seq_len,
                        activation_recomputation=2,
                        log_level="CRITICAL",
                        first_embedding=True if stage == 0 else False,
                        last_embedding=True if stage == pp_size-1 else False,
                        model_config=model_config
                    )
                    if training_data['memory_left'] < 0: 
                        break
                    else:
                        optimal_layers = layer
                if optimal_layers > 0:
                    gpu.max_layers[f'{pp_size}_{stage}'] = optimal_layers
                    # gpu.total_cost[f'{pp_size}'] = latency * gpu.cost * model.dp_size * model.tp_size / 3600 #gpu.cost 单位是 $/h 而 latency单位是s

