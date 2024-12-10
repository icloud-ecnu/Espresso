"""
一个使用CoLLie对LLaMA基座进行全参量Instruct Tuning，从而得到Alpaca的实例。
"""
import os
import shutil
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
parser = argparse.ArgumentParser(description='Enumerate model layer partitions for GPUs.')
parser.add_argument('--partitions', type=str, default="0,8,19,29", help='A comma-separated list of numbers representing parts_partition_arr (e.g., "0,5,16,29").')
parser.add_argument('--gpuType', type=str, default="A100", help='GPU type')
parser.add_argument('--syn',  action='store_true', help='GPU synchronize engine.py')
parser.add_argument('--baseDir', type=str, default="partition_data", help='data dir')
parser.add_argument('--seqlen', type=int, default=1400, help='seq len')
parser.add_argument('--batch', type=int, default=16, help='batch size')
parser.add_argument('--modelName', type=str, default="llama_3b", help='model name')
parser.add_argument('--fwdBwdLists', type=str,default=None, help='model name')
parser.add_argument('--gradient_accumulation_steps', type=int,default=4, help='gradient_accumulation_steps')
parser.add_argument('--dpSize', type=int,default=1, help='dpSize')

args = parser.parse_args()




# 从参数中获取 parts_partition_arr 值
parts_partition_arr = list(map(int, args.partitions.split(',')))
configTag = args.gpuType


# Convert integer list to string list and then join with '_'
str_list = map(str, parts_partition_arr)
joined_str = '_'.join(str_list)

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上回退三级目录到项目目录
project_dir = current_dir
for _ in range(3):
    project_dir = os.path.dirname(project_dir)

rank = os.environ.get("RANK")
# 设置basedir目录
basedir = os.path.join(project_dir, 'collie', 'examples', 'alpaca', args.baseDir, f"{args.gpuType}", f"{rank}_{configTag}_{joined_str}")

tmpfilepath = basedir + "_tmp"
os.environ['FILE_NAME'] = tmpfilepath
os.environ['SYN'] = "True" if args.syn else "False"


if os.path.exists(tmpfilepath):
    shutil.rmtree(tmpfilepath)
os.makedirs(tmpfilepath)


import deepspeed
from deepspeed.utils import logger
import deepspeed.runtime.utils as ds_utils
from deepspeed.runtime.pipe.module import LayerSpec
from torch import nn

def _partition_layers(self, method='uniform'):
    num_stages = self._topo.get_dim('pipe')
    stage_id = self._topo.get_coord(self.global_rank).pipe
    if self.global_rank == 0:
        logger.info(f'Partitioning pipeline stages with method {method}')
    method = method.lower()
    # Each stage gets a simple uniform number of layers.
    if method == 'uniform':
        num_layers = len(self._layer_specs)
        self.parts = ds_utils.partition_uniform(num_items=num_layers,
                                                num_parts=num_stages)
    elif method == 'zqn':
        num_layers = len(self._layer_specs)
        self.parts = parts_partition_arr
    elif method == 'parameters':
        param_counts = self._count_layer_params()
        self.parts = ds_utils.partition_balanced(weights=param_counts,
                                                num_parts=num_stages)
        print(self.parts)
    elif method.startswith('type:'):
        layertype = method.split(':')[1]
        binary_weights = [0] * len(self._layer_specs)
        for idx in self._find_layer_type(layertype):
            binary_weights[idx] = 1
        self.parts = ds_utils.partition_balanced(weights=binary_weights,
                                                num_parts=num_stages)
    elif method == 'profile':
        raise NotImplementedError(f'Partitioning method {method} not implemented.')
    else:
        raise NotImplementedError(f'Partitioning method {method} not implemented.')

    # Print some information on the partitioning.
    if self.global_rank == 0:
        for stage in range(num_stages):
            start = self.parts[stage]
            stop = self.parts[stage + 1]
            print(f'stage={stage} layers={stop - start}')
            for idx, layer in enumerate(self._layer_specs[start:stop]):
                name = str(layer)
                if isinstance(layer, LayerSpec):
                    name = layer.typename.__name__
                if isinstance(layer, nn.Module):
                    name = layer.__class__.__name__
                else:
                    try:
                        name = layer.__name__
                    except AttributeError:
                        pass
                print(f'    {idx + start:2d}: {name}')
        if self.loss_fn:
            try:
                print(f'  loss: {self.loss_fn.__name__}')
            except AttributeError:
                print(f'  loss: {self.loss_fn.__class__.__name__}')

    self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

deepspeed.runtime.pipe.module.PipelineModule._partition_layers=_partition_layers
import sys
sys.path.append('../../')
import json
import torch
os.environ['CUDA_DEVICE_MAX_CONNECTIONS ']='1'
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig

from collie.config import CollieConfig

from collie.data import CollieDatasetForTraining
from collie.data import CollieDataLoader

from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity, EvaluatorForGeneration

# from collie.models.llama.model import LlForCausalLM
from collie import CollieConfig, LlamaForCausalLM, DashProvider, LossMonitor,EvalMonitor,StepTimeMonitor , TGSMonitor, MemoryMonitor, LRMonitor,NetworkIOMonitor, CollieDatasetForTraining, CollieDatasetForGeneration, EvaluatorForGeneration, PrunedLlamaForCausalLM
from collie.metrics import DecodeMetric, PPLMetric, BleuMetric
from collie.module import GPTLMLoss
# OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=4,3,1,0 torchrun --standalone --nproc_per_node=4 train.py --partitions 0,7,17,23,29
# 1. 设置路径
# 1.1 预训练模型路径
pretrained_model = os.path.join(project_dir, 'model', args.modelName)
tokenizer_path = os.path.join(project_dir, 'model', 'open_llama_3b_v2')
# 1.2 数据集路径
# data_path = 'alpaca.json'
data_path = 'alpaca_evol_instruct_70k.json'

# 1.3 Eval的decode结果保存路径
save_path = './result'
config = CollieConfig.from_pretrained(pretrained_model)
# 2.2 添加配置
config.pp_partition_method = 'zqn'
config.dp_size = args.dpSize
config.tp_size = 1
config.pp_size = len(parts_partition_arr) - 1
config.train_micro_batch_size = args.batch
config.use_cache=False
config.eval_batch_size = 1
config.gradient_accumulation_steps = args.gradient_accumulation_steps # real_batch = train_micro_batch_size * gradient_accumulation_steps
config.eval_per_n_epochs = 1
config.train_epochs = 1
config.ds_config = {
    "bf16": {
        "enabled": True
    },
    "flops_profiler": {
        "enabled": True,
        "profile_step": 50,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": True,
        "output_file": "profile.log"
    },
    "monitor_config": {
        "enabled": True,
        "tag":configTag,
        # "wandb": {
        #     "enabled": True,
        #     "team": "echozhou1010",
        #     "project": "LLM-expr2",
        #     "group": "summary"
        # },
        "csv_monitor": {
                "enabled": True,
                "output_path": "./ds_logs/"
        },
    },
}

## 
# if os.environ.get("RANK") == '0':
#     torch.cuda.memory._record_memory_history(True, True, True, 0, True)


from deepspeed.runtime.pipe import schedule
def cal_op(micro_batches, num_stages, stage_id):
    _INSTRUCTION_MAP = {
        schedule.ForwardPass: "F",
        schedule.BackwardPass: "B",
    }
    pipe_schedule = schedule.TrainSchedule(micro_batches=micro_batches,stages=num_stages,stage_id=stage_id)
    a = []
    for step_cmds in pipe_schedule:
        # For each instruction in the step
        for cmd in step_cmds:
            if type(cmd) not in _INSTRUCTION_MAP:
                continue
            a.append(_INSTRUCTION_MAP[type(cmd)])
    return a

fwdBwdLists = args.fwdBwdLists
if not fwdBwdLists:
    tmp = ["F","F","F","F","B","B","B","B"]
    ans = [tmp for x in range(config.pp_size)]
    # ans.append(['F','B','F','B','F','B','F','B'])
    # ans = []
    # for stage_id in range(config.pp_size):
    #     tmp = cal_op(args.gradient_accumulation_steps,config.pp_size,stage_id)
    #     ans.append(tmp)
    fwdBwdLists = str(ans)
print(f"fwdBwdLists = {fwdBwdLists}")
os.environ["fwdBwdLists"] = str(fwdBwdLists)


# if os.environ.get("RANK") == '0':
#     # save a snapshot of the memory allocations
#     s = torch.cuda.memory._snapshot()
#     with open(f"snapshot.pickle", "wb") as f:
#         dump(s, f)

#     # tell CUDA to stop recording memory allocations now
#     torch.cuda.memory._record_memory_history(enabled=None)

# import ipdb
# import os
# if os.environ.get("RANK") == '3':
#     ipdb.set_trace()

config.seed = 1024
# 3. 设置tokenizer
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, add_eos_token=False)

# 4. 加载数据集
data_num = config.train_micro_batch_size * 40
dataset = CollieDatasetForTraining.from_json(data_path, tokenizer=tokenizer,max_length=args.seqlen) 
# print(f"len = {len(dataset)}")
train_dataset = dataset[:640]

model = LlamaForCausalLM.from_config(config=config)




from collections import defaultdict
# 创建一个默认字典来存储每个设备上的参数数量
device_param_count = defaultdict(int)

# 遍历模型的所有参数
for name, param in model.named_parameters():
    # 获取参数所在的设备
    device = param.device
    # 统计该设备上的参数数量
    device_param_count[str(device)] += param.numel()

# 打印出每个设备上的参数数量
for device, param_count in device_param_count.items():
    print(f"Number of parameters on {device}: {param_count}")

# with open(f'/home/ecnu/echozhou/collie/examples/alpaca/model_parameters', 'a+', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(model_parameters)

# print(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
import transformers

# 7. 添加监视器
monitors = [
    StepTimeMonitor(config),
    TGSMonitor(config),
    MemoryMonitor(config),
    LossMonitor(config),
    # EvalMonitor(config),
]

# # 8. 添加Evaluator
# evaluator_ppl = EvaluatorForPerplexity(
#     model = model,
#     config = config,
#     dataset = eval_dataset,
#     loss_fn = GPTLMLoss(-100),
#     monitors = [
#         EvalMonitor(config)
#     ],
#     metrics = {
#         'ppl': PPLMetric()
#     }
# )
filepath = os.path.join(tmpfilepath,"memoryCost.txt")
if os.environ.get("RANK") == '0':
    with open(filepath, 'a') as file:
        file.write(f"模型加载之后，max allocated:{torch.cuda.max_memory_allocated()} memory_allocated:{torch.cuda.memory_allocated()} memory_reserved={torch.cuda.memory_reserved()}\n")

# 9. 实例化trainer
trainer = Trainer(
    model=model,
    loss_fn=GPTLMLoss(-100),
    tokenizer=tokenizer,
    # lr_scheduler=lr_scheduler,
    config=config,
    optimizer=optimizer,
    train_dataset = train_dataset,
    monitors = monitors,    
    # evaluators = [evaluator_ppl]
)

if os.environ.get("RANK") == '0':
    with open(filepath, 'a') as file:
        file.write(f"实例化trainer之后，max allocated:{torch.cuda.max_memory_allocated()} memory_allocated:{torch.cuda.memory_allocated()} memory_reserved={torch.cuda.memory_reserved()}\n")

# 10. 训练/验证
trainer.train()
rank = os.environ.get("RANK")
max_mem = torch.cuda.max_memory_allocated() 
sys.stdout.flush()


print(f"最后 rank = {rank}   Max memory allocated on GPU: {max_mem / 1024**3:.2f} GB\n")
print_statement = f"rank = {rank}   Max memory allocated on GPU: {max_mem / 1024**3:.2f} GB\n"
with open(filepath, 'a') as file:
    file.write(print_statement)

"""
16*1400*3200
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,4 torchrun --standalone --nproc_per_node=3 train.py                         
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,4 torchrun --standalone --nproc_per_node=3 train.py --partitions 0,11,18,29
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,4,0,1 torchrun --standalone --nproc_per_node=4 train.py --partitions 0,9,18,23,29 --syn

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,2,3,4 torchrun --standalone --nproc_per_node=4 train.py  --partitions 0,5,11,18,29

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,4 torchrun --standalone --nproc_per_node=3 train.py --partitions 0,10,19,29 --gpuType sequence_fix_A100 --syn

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,1,0 torchrun --standalone --nproc_per_node=4 train.py --partitions 0,9,18,23,29 --syn --gpuType test --syn

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,2,3,4,1 torchrun --standalone --nproc_per_node=5 train.py --partitions 0,1,9,18,27,29 --syn --gpuType sequence_fix_mix --syn
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,4,0 torchrun --standalone --nproc_per_node=4 train.py --partitions 0,11,18,28,29 --syn --gpuType sequence_fix_mix --syn
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,2,3,4 torchrun --standalone --nproc_per_node=4 train.py --partitions 0,1,11,18,29 --syn --gpuType sequence_fix_mix --syn

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,0,3 torchrun --standalone --nproc_per_node=3 train.py --partitions 0,9,18,29 --syn --gpuType sequence_fix_mix_batch8 --syn

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,0,4 torchrun --standalone --nproc_per_node=4 train.py --partitions 0,11,22,26,29 --syn --gpuType iSomer --syn --seqlen 1800 --baseDir ./experiments/expr2/real_data

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=4,0,2,3 torchrun --standalone --nproc_per_node=4 train.py --partitions 0,5,10,19,29 --syn --gpuType sequence_fix_mix --syn

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,0 torchrun --standalone --nproc_per_node=3 train.py --partitions 0,12,23,29 --syn --gpuType iSomer --syn --seqlen 1700 --baseDir ./experiments/expr2/real_data
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,0 torchrun --standalone --nproc_per_node=3 train.py --partitions 0,10,19,23,29 --syn --gpuType iSomer --syn --seqlen 1500 --baseDir ./experiments/expr2/real_data


OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,4,0,1 torchrun --standalone --nproc_per_node=5 train.py --partitions 0,5,10,23,26,29 --syn --gpuType sequence_fix_mix --syn --seqlen 1700 --baseDir ./experiments/expr2/real_data

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node=3 train.py --partitions 0,10,19,23,29 --syn --gpuType iSomer --syn --seqlen 1500 --baseDir ./experiments/expr2/real_data

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,4 torchrun --standalone --nproc_per_node=4 train.py --partitions 0,5,7,18,29 --syn --gpuType test --syn --seqlen 1400 --baseDir ./experiments/expr2/real_data

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2,3,4,0,1 torchrun --standalone --nproc_per_node=5 train.py --partitions 0,5,10,23,26,29 --syn --gpuType sequence_fix_mix --syn --seqlen 1700 --baseDir ./experiments/expr2/real_data

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=3,2,4,0,1 torchrun --standalone --nproc_per_node=5 train.py --partitions 0,4,13,22,25,29 --syn --gpuType sequence_fix_mix --syn --seqlen 100 --baseDir ./experiments/expr2/real_data

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 train.py --partitions 0,4,10,18,25,29 --syn --gpuType sequence_fix_mix --syn --seqlen 1400 --baseDir ./experiments/expr2/real_data

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=3,4 torchrun --standalone --nproc_per_node=2  test.py --modelpath /home/ecnu/Espresso/llm/model/open_llama_3b_v2 --partitions 0,17,35 --syn --gpuType test --syn --seqlen 1000 --baseDir ./useTest 

 55;31;18'
train.py --partitions 0,10,18,27 --syn --gpuType motivation1_3A10G --syn --seqlen 1600 --baseDir ./experiments/motivation/real_data

OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=3,4 torchrun --standalone --nproc_per_node=2  test.py --modelpath /home/ecnu/Espresso/llm/model/open_llama_3b_v2 --partitions 0,17,35 --syn --gpuType test --syn --seqlen 1000 --baseDir ./useTest 
"""
