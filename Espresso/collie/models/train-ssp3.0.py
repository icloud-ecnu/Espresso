import sys
import argparse
from tkinter.tix import Tree
sys.path.append('../../')
import json
import torch
import os
os.environ['CUDA_DEVICE_MAX_CONNECTIONS ']='1'
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from collie.models.llamakd.model import LlamaKDLayer
from collie.config import CollieConfig

from collie.data import CollieDatasetForTraining, CollieKDDatasetForTraining
from collie.data import CollieDataLoader

from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity, EvaluatorForGeneration
from collie.callbacks import CheckpointCallback
from collie.models.llama.model import LlamaForCausalLM
from collie import CollieConfig, LlamaForCausalLM, DashProvider, LossMonitor,EvalMonitor,StepTimeMonitor , TGSMonitor, MemoryMonitor, LRMonitor, CollieDatasetForTraining, CollieDatasetForGeneration, EvaluatorForGeneration, LlamaKDForCausalLM
from collie.metrics import DecodeMetric, PPLMetric, BleuMetric
from collie.module import GPTKDLMLoss,GPTLMLoss
args = argparse.ArgumentParser()
import huggingface_hub
huggingface_hub.login('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')
# args.add_argument("--model_name", type=str, default='/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf')
args.add_argument("--model_name", type=str, default='/mnt/petrelfs/leizhikai/llama2-7b')
args.add_argument("--train_data_path", type=str, default='/mnt/petrelfs/leizhikai/data/train')
args.add_argument("--eval_data_path", type=str, default='/mnt/petrelfs/leizhikai/data/eval')
args.add_argument("--use_flash", type=int, default=1)
args.add_argument("--lr", type=float, default=5e-5)
args.add_argument("--dp_size", type=int, default=1)
args.add_argument("--tp_size", type=int, default=1)
args.add_argument("--pp_size", type=int, default=8)
args.add_argument("--train_epochs", type=int, default=1)
args.add_argument("--train_micro_batch_size", type=int, default=8)
args.add_argument("--gradient_accumulation_steps", type=int, default=4)
args = args.parse_args()

save_path = './result'

callbacks = [CheckpointCallback("s3://P_model_weights/leizhikai/kd-ssp-fix3.0",protocol="petrel",
                                every_n_batches=3200, model_only=False,peft_only=False)]
config = CollieConfig.from_pretrained(args.model_name)
# config.pp_partition_method = 'uniform'
# 2.2 添加配置
config.lr = args.lr
config.dp_size = args.dp_size
config.tp_size = args.tp_size
config.pp_size = args.pp_size
config.train_micro_batch_size = args.train_micro_batch_size
config.use_cache=False
config.eval_batch_size = 1
config.gradient_accumulation_steps = args.gradient_accumulation_steps
config.eval_per_n_steps = 400
config.train_epochs = args.train_epochs

config.seed = 1024

tokenizer = LlamaTokenizer.from_pretrained(args.model_name, add_eos_token=False)
from  clm_tools_pile import get_pile_for_perplexity

train_dataset = CollieKDDatasetForTraining.from_json(args.train_data_path, tokenizer=tokenizer,max_length=2048)
eval_dataset = CollieKDDatasetForTraining.from_json(args.eval_data_path, tokenizer=tokenizer,max_length=2048)
total_step = int(len(train_dataset)*config.train_epochs/(config.dp_size*config.train_micro_batch_size*config.gradient_accumulation_steps))

config.ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "monitor_config": {
        "enabled": True,
        "tag":"test",
        "wandb": {
            "enabled": True,
            "team": "icalk",
            "project": "supervised-finetuning",
            "group": "A800"
        }
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": config.lr,
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
            "torch_adam": True,
        }
  },
#   "scheduler": {
#     "type": "WarmupDecayLR",
#     "params": {
#       "warmup_min_lr": 0,
#       "warmup_max_lr": config.lr,
#       "warmup_num_steps": int(total_step*0.03),
#       "warmup_type":"linear",
#       "total_num_steps": total_step
#     }
#   },
}

if os.environ.get("RANK")==0:
    print("I a,m rank 0")
model = LlamaKDForCausalLM.from_pretrained(args.model_name, config=config)




def change_forward(model, k=256,tp="bf16"):
    import torch
    import types
    import torch.nn.functional as F
    import numpy as np
    from einops import rearrange
    from typing import Union, Optional, Tuple
    from collie.models.utils import flash_attention
    import math
    def forward(self, input):
        return F.linear(input, self.weight*self.mask, self.bias)

    def modify_ffn(ffn, path):
        keep = torch.load(path)
        ffn.mask = torch.zeros(ffn.weight.shape,device=ffn.weight.device).to(torch.bfloat16)
        ffn.mask[keep,:]=1
        ffn.weight.data *= ffn.mask
        ffn.forward = types.MethodType(forward, ffn)
        
    # encoder
    ind = 0
    for i,mm in enumerate(model):
        from collie.utils import env
        if os.environ.get("RANK")=='0':
            pass
            # print(mm,type(mm))
            # print("*"*66)
        if type(mm)==LlamaKDLayer:
            pp_rank = env.pp_rank
            layer_idx = pp_rank*4+ind
            ind += 1
            if os.environ.get("RANK")=='1':
                print("lalalal",layer_idx)
            # print(dir(model._layer_specs[i].typename.get_submodule()))
            path = os.path.join("/mnt/petrelfs/leizhikai/moe/Llama-2-7b/", 'param_fixed', 'layer'+str(layer_idx))
            ffn = mm.llamalayers[0].mlp["gate_proj"]
            modify_ffn(ffn, path) 

            ffn = mm.llamalayers[0].mlp["up_proj"]
            modify_ffn(ffn, path) 
    assert ind==4
# change_forward(model,k=128)
change_forward(model.forward_funcs,k=64,tp="bf16")
# print(model)
params = []

for name, param in model.named_parameters():
    if name.count(".1.") ==1 and name.count(".0.")==0:
        if os.environ.get("RANK")=='0':
            print(name)
        # param.require_grad = False
    elif  name.count(".1.") ==1:
        assert name.count(".0.")==1
        if int(name.split('.')[2])==0:
            # param.require_grad = False
            if os.environ.get("RANK")=='0':
                print(name)
        else:
            params += [param]
    elif name.count(".1.") ==2:
        # param.require_grad = False
        if os.environ.get("RANK")=='0':
            print(name)
    else:
        params += [param]

params = [{'params': params}]
optimizer = torch.optim.AdamW(params, lr=config.lr)

import transformers
lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_step*0.03), num_training_steps=total_step)

# if os.environ.get('RANK')=='0':
#     for name, param in model.named_parameters():
#         print(name, param.requires_grad)
# 7. 添加监视器
monitors = [
    StepTimeMonitor(config),
    TGSMonitor(config),
    MemoryMonitor(config),
    LossMonitor(config),
    EvalMonitor(config),
    LRMonitor(config)
]


# save_call = CheckpointCallback(max=10,)
# 8. 添加Evaluator
evaluator_ppl = EvaluatorForPerplexity(
    model = model,
    config = config,
    dataset = eval_dataset,
    loss_fn = GPTKDLMLoss(-100,mode='eval'),
    monitors = [
        EvalMonitor(config)
    ],
    metrics = {
        'ppl': PPLMetric()
    }
)

# 9. 实例化trainer
trainer = Trainer(
    model=model,
    loss_fn=GPTKDLMLoss(-100),
    tokenizer=tokenizer,
    lr_scheduler=lr_scheduler,
    config=config,
    optimizer=optimizer,
    train_dataset = train_dataset,
    monitors = monitors,
    evaluators = [evaluator_ppl],
    callbacks=callbacks
)

# if os.environ.get("RANK"):
    # print("!!!!!!!!!!!!!!!!!!!!",len(trainer.engine.forward_funcs))
# change_forward(trainer.engine.forward_funcs,k=64,tp="bf16")
# 10. 训练/验证
trainer.train()

# OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py    