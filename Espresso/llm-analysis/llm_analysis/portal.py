
import argparse
import json
import os.path
from pathlib import Path
from collections import defaultdict
import fire
from llm_analysis.pipeline_model import cal_pipeline_time
from llm_analysis.pipeline_model import cal_op
import llm_analysis.analysis
from llm_analysis.utils import _latency_to_string, _num_to_string, within_range
# python portal.py --model_name open_llama_3b_v2 --gpu_name a100-pcie-40gb --tp_size 1 --pp_size 3 --sp_size 1 --dp_size 1 --gradient_accumulation_steps 4 -b 16 --seq_len 1400  --total_num_tokens 1280 --activation_recomputation 2 --flops_efficiency 0.43 --hbm_memory_efficiency 0.55 --mlp_recompute_gelu True  --output_dir ./data
# step1 启动llm-analysis
def read_json(filename):
    with open(filename, "r") as file:
        gpu_efficiencies = json.load(file)
    return gpu_efficiencies

def _calculate_num_microbatch(sequence):
    num_microbatch = 0
    for char in sequence:
        if char == 'F':
            num_microbatch += 1
        else:
            break  # 当遇到非 'F' 的字符时，停止计数
    return num_microbatch


def train(
    model_name: str,
    partitions: str,
    gpu_name: str,
    tp_size: int,
    pp_size: int,
    sp_size: int,
    dp_size: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    seq_len: int,
    output_dir: str,
    activation_recomputation: int = 2,
    breakdown: bool = False,
    model_layer: int = None,
    useNew: str = False,
    model_config: dict = None,
    ds_zero: int = 0,
    flash_attn: bool = True,
):
    """
        训练函数
        参数:
            model_name (str): 模型名称
            gpu_name (str): GPU名称列表
            partition (str): 分区列表
            tp_size (int): TP大小
            pp_size (int): PP大小
            sp_size (int): SP大小
            dp_size (int): DP大小
            gradient_accumulation_steps (int): 梯度累积步数
            batch_size (int): 批量大小
            seq_len (int): 序列长度
            activation_recomputation (int, optional): 激活重新计算，默认值为2
            output_dir (str): 输出目录
    """
    # 在这里实现训练逻辑，使用上述参数来配置训练过程
    # print(f"batch_size = {batch_size}")
    gpu_name = gpu_name.split(";")
    partitions = [int(p) for p in partitions.split(";")]
    num_stages = len(gpu_name)
    assert num_stages == pp_size
    fwd_time = []
    comm_time = []
    breakdownInfo = defaultdict(dict)
    out_of_memory = 0
    isOK = True
    activation_mem = 0
    # print(num_stages,partitions)
    networkBandwidth = 2.5 * (1024 ** 3)
    # networkBandwidth = 5 * (1024 ** 3)
    allreduce = []
    # print(f"dpsize = {dp_size}")
    # print(partitions)
    for i in range(num_stages):
        # 获取gpu性能信息
        gpu_efficiencies = read_json(f"{Path(__file__).parent}/gpu_efficiency.json")
        gpu_info = gpu_efficiencies.get(gpu_name[i])
        if gpu_info:
            flops_efficiency = gpu_info['flops_efficiency']
            hbm_memory_efficiency = gpu_info['hbm_memory_efficiency']
        else:
            raise ValueError(f"No data available for GPU type: {gpu_name[i]}")

        num_layers_per_gpu = partitions[i]
        first_embedding = True if i == 0 else False
        last_embedding = True if i + 1 == num_stages else False
        sequence = cal_op(gradient_accumulation_steps,num_stages,i)
        num_microbatch = _calculate_num_microbatch(sequence)
        # import time
        # start = time.time()
        information = llm_analysis.analysis.train(
                    model_name=model_name, gpu_name=gpu_name[i], tp_size=tp_size, 
                    pp_size=1, dp_size=dp_size, sp_size=sp_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    batch_size_per_gpu=batch_size,
                    num_layers_per_gpu=num_layers_per_gpu, num_microbatch=num_microbatch,
                    flops_efficiency=flops_efficiency, 
                    hbm_memory_efficiency=hbm_memory_efficiency,
                    seq_len=seq_len, activation_recomputation=2, log_level="CRITICAL",
                    first_embedding=first_embedding,last_embedding=last_embedding,
                    output_dir=output_dir,
                    model_layer=model_layer,
                    model_config = model_config,
                    ds_zero=ds_zero,
                    flash_attn=flash_attn,
        )
        # print("i = ",i," information = ",information,f" num_layers_per_gpu={num_layers_per_gpu} {information['weight_memory_layer']}  {ds_zero}")
        comm_size = information['weight_memory_layer'] * 8 * num_layers_per_gpu
        # print(comm_size)
        if i == 0 or i == num_stages - 1:
            comm_size += information['weight_memory_embedding'] * 8
            # print(information['weight_memory_embedding'] * 8)
        if dp_size > 1:
            allreduce.append(comm_size / networkBandwidth)
        else:
            allreduce.append(0)
        # end = time.time()
        # print(f"portal i = {i} time cost = {end - start}")
        # print("rank = ",i," info = ",information)
        # return result.stdout, result.stderr, result.returncode
        # print(i,information['memory_left'],information['total_memory'],partitions,gpu_name)
        # print(gpu_name[i],num_layers_per_gpu,num_microbatch)
        memory_left = information['memory_left']
        # print(f"comm_latency = {information['comm_latency']}")
        if memory_left < 0:
            # print(f"i = {i} mem = {information['total_memory']} num_microbatch = {num_microbatch} num_layers_per_gpu = {num_layers_per_gpu}")
            out_of_memory += -memory_left
            isOK = False
        memory_cost = information['total_memory']
        latency_cost = information['total_latency']
        breakdownInfo[str(i)]['mem'] = memory_cost
        breakdownInfo[str(i)]['mem_left'] = memory_left
        breakdownInfo[str(i)]['latency'] = latency_cost
        breakdownInfo[str(i)]['activation_memory'] = information['activation_memory_per_layer'] * num_layers_per_gpu
        fwd_time.append(latency_cost)
        comm_time.append(information['comm_latency'])
        # print(f"info = {information}")
        # print(_num_to_string(memory_cost))
        # print(_latency_to_string(latency_cost))
    
    if not isOK:
        out_of_memory = out_of_memory / 1024 / 1024 + 500
        if breakdown:
            return False,out_of_memory,breakdownInfo
        else:
            return False,out_of_memory
    # print(fwd_time)
    bwd_time = [x*3 for x in fwd_time]
    # print(comm_time)
    comm_time = [x*(num_stages - 1)*0.8 for x in comm_time]
    # print(f"comm_time = {comm_time}")
    # print(comm_time)
    for i in range(num_stages):
        # bwd_time[i] += comm_time[i]
        fwd_time[i] += comm_time[i]
    # print(fwd_time)
    # print(f"comm_latency = {allreduce}")
    # print(f"fwd_time = {fwd_time}")
    # print(f"bwd_time = {bwd_time}")
    # print(fwd_time)
    startTime,time,op = cal_pipeline_time(fwd_time,bwd_time,gradient_accumulation_steps,need=True,useNew = useNew)
    # print(time)
    comm_latency = 0
    latency = 0
    for i in range(num_stages):
        for j in range(len(time[i])):
            latency = max(latency,time[i][j] + allreduce[i])
    comm_latency = latency - time[0][-1]
    # print(f"allreduce = {allreduce}")
    breakdownInfo.update({"comm_latency":comm_latency})
    # comm latency???

    
    # print(partitions)
    # print(f"latency = {latency}")
    # print(breakdownInfo)
    if breakdown:
        return True,latency,breakdownInfo
    # print(latency)
    return True,latency
    


if __name__ == "__main__":
    fire.Fire(serialize=lambda x: json.dumps(x, indent=4))



"""
python portal.py train \
    --model_name "open_llama_3b_v2" \
    --gpu_name "a100-pcie-40gb;a100-pcie-40gb;a100-pcie-40gb" \
    --partitions "9;9;8" \
    --tp_size 1 \
    --pp_size 3 \
    --sp_size 1 \
    --dp_size 1 \
    --gradient_accumulation_steps 4 \
    --batch_size 16 \
    --seq_len 1400 \
    --activation_recomputation 2 \
    --output_dir "./testData"

python portal.py train \
    --model_name "open_llama_3b_v2" \
    --gpu_name "a100-pcie-40gb;a100-pcie-40gb;4090-pcie-24gb;a100-pcie-40gb" \
    --partitions "10;11;4;1" \
    --tp_size 1 \
    --pp_size 4 \
    --sp_size 1 \
    --dp_size 1 \
    --gradient_accumulation_steps 4 \
    --batch_size 16 \
    --seq_len 1800 \
    --activation_recomputation 2 \
    --output_dir "./testData"

0;9;18;23;29
a100-pcie-40gb;a100-pcie-40gb;4090-pcie-24gb;4090-pcie-24gb

python -m analysis train --model_name open_llama_3b_v2 --gpu_name a100-pcie-40gb --tp_size 1 --pp_size 3 --sp_size 1 --dp_size 1 --gradient_accumulation_steps 4 -b 16 --seq_len 1400  --total_num_tokens 1280 --activation_recomputation 2 --flops_efficiency 0.43 --hbm_memory_efficiency 0.55 --mlp_recompute_gelu True --output_dir ./data --output_detail_file_suffix test

python -m analysis train --model_name open_llama_3b_v2 --gpu_name a100-pcie-40gb                 --tp_size 1 --pp_size 1 --sp_size 1 --dp_size 1 --gradient_accumulation_steps 4                 -b 16 --seq_len 1400 --activation_recomputation 2 --flops_efficiency 0.44                 --num_layers_per_gpu 0 --first_embedding True --last_embedding False --num_microbatch 4                --hbm_memory_efficiency 0.52 --output_dir ./testData
python -m llm_analysis.llm_analysis.portal train                 --model_name "open_llama_3b_v2"                 --gpu_name "a100-pcie-40gb;a100-pcie-40gb;4090-pcie-24gb;4090-pcie-24gb"                 --partitions "0;13;21;24;29"                 --tp_size 1                 --pp_size 4                 --sp_size 1                 --dp_size 1                 --gradient_accumulation_steps 4                 --batch_size 16                 --seq_len 1400                 --activation_recomputation 2                 --output_dir "./testData"
"""


