from pathlib import Path
from collections import defaultdict
import json
import llm_analysis.analysis
from llm_analysis.pipeline_model import cal_pipeline_time

GPU_CONFIG_FILE = "gpu_config.json"

class GPU:
    def __init__(self, name, cost, total_cost, flops_efficiency, hbm_memory_efficiency):
        self.name = name
        self.max_layers = defaultdict(int)
        self.cost = cost
        self.total_cost = total_cost
        self.flops_efficiency = flops_efficiency
        self.hbm_memory_efficiency = hbm_memory_efficiency

class Plan:
    def __init__(self, cost, gpu_counts, pp_size, dp_size, tp_size):
        self.cost = cost
        self.gpu_counts = gpu_counts
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.tp_size = tp_size

def load_gpus_from_config():
    with open(f"{Path(__file__).parent}/{GPU_CONFIG_FILE}", 'r') as file:
        data = json.load(file)
        gpus = []
        for name, attributes in data.items():
            gpu = GPU(name, 
                      cost=attributes['cost'],
                      total_cost = 0,
                      flops_efficiency=attributes['flops_efficiency'],
                      hbm_memory_efficiency=attributes['hbm_memory_efficiency'])
            gpus.append(gpu)
        return gpus

def test_gpu_performance(model_name, gpus, global_batch_size, gradient_accumulation_steps, seq_len, slo, PP_SIZES, DP_SIZES, TP_SIZES, alph=2.7):
    for gpu in gpus:
        for pp_size in PP_SIZES:
            for dp_size in DP_SIZES:
                for tp_size in TP_SIZES:
                    if global_batch_size % (dp_size * gradient_accumulation_steps) != 0: continue
                    l, r, ans, latency = 1, 1000, -1, 0
                    while l <= r:
                        mid = (l + r) // 2
                        data = llm_analysis.analysis.train(model_name=model_name, gpu_name=gpu.name, tp_size=tp_size, pp_size=pp_size, dp_size=dp_size,
                                                           gradient_accumulation_steps=gradient_accumulation_steps, global_batch_size=global_batch_size,
                                                           batch_size_per_gpu=global_batch_size // (dp_size * gradient_accumulation_steps), num_layers_per_gpu=mid,
                                                           num_microbatch=0, flops_efficiency=gpu.flops_efficiency, hbm_memory_efficiency=gpu.hbm_memory_efficiency,
                                                           batch_size=16, seq_len=seq_len, activation_recomputation=2)
                        time_cost = data['total_latency']
                        fwd_time = [time_cost for _ in range(pp_size)]
                        bwd_time = [x * alph for x in fwd_time]
                        if data['memory_left'] < 0 or cal_pipeline_time(fwd_time, bwd_time, gradient_accumulation_steps) > slo:
                            r = mid - 1
                        else:
                            l = mid + 1
                            ans = mid
                            latency = time_cost
                    if ans > 0:
                        gpu.max_layers[f'{pp_size}_{dp_size}_{tp_size}'] = ans
                        gpu.total_cost = latency * gpu.cost


                       
