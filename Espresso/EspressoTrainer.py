import os
import shutil
import argparse
import deepspeed
import sys
import json
import torch
from torch import nn
import deepspeed
from deepspeed.utils import logger
import deepspeed.runtime.utils as ds_utils
from deepspeed.runtime.pipe.module import LayerSpec
from transformers import LlamaTokenizer
from transformers.generation.utils import GenerationConfig
from collections import defaultdict

from collie.config import CollieConfig
from collie.data import CollieDatasetForTraining
from collie.controller.trainer import Trainer
from collie.module import GPTLMLoss
from collie import CollieConfig, LlamaForCausalLM, DashProvider, LossMonitor,EvalMonitor,StepTimeMonitor , TGSMonitor, MemoryMonitor, LRMonitor,NetworkIOMonitor, CollieDatasetForTraining, CollieDatasetForGeneration, EvaluatorForGeneration, PrunedLlamaForCausalLM

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
        self.parts = eval(os.environ.get('parts_partition_arr'))
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



class EspressoTrainer:
    def __init__(self, args):
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.args = args
        self._setup_directories()
        self.config = self._load_config()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.optimizer = self._configure_optimizer()
        self.train_dataset = self._load_dataset()

    def _setup_directories(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = current_dir
        rank = os.environ.get("RANK", "0")
        parts_partition_arr = list(map(int, self.args.partitions.split(',')))
        configTag = self.args.gpuType
        joined_str = '_'.join(map(str, parts_partition_arr))
        basedir = os.path.join(self.project_dir, 'data', self.args.baseDir, f"{self.args.gpuType}", f"{rank}_{configTag}_{joined_str}")
        self.tmpfilepath = basedir + "_tmp"
        os.environ['FILE_NAME'] = self.tmpfilepath
        os.environ['SYN'] = "True" if self.args.syn else "False"
        if os.path.exists(self.tmpfilepath):
            shutil.rmtree(self.tmpfilepath)
        os.makedirs(self.tmpfilepath)

    def _load_config(self):
        pretrained_model = self.args.modelpath
        config = CollieConfig.from_pretrained(pretrained_model)
        parts_partition_arr = list(map(int, self.args.partitions.split(',')))
        # Configurations
        config.pp_partition_method = 'zqn'
        config.dp_size = self.args.dpSize
        config.tp_size = 1
        config.pp_size = len(parts_partition_arr) - 1
        config.train_micro_batch_size = self.args.batch
        config.use_cache = False
        config.eval_batch_size = 1
        config.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        config.eval_per_n_epochs = 1
        config.train_epochs = self.args.train_epochs
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
                "tag": self.args.gpuType,
                "csv_monitor": {
                    "enabled": True,
                    "output_path": "./ds_logs/"
                },
            },
        }
        return config

    def _load_tokenizer(self):
        tokenizer_path = self.args.modelpath
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, add_eos_token=False)
        return tokenizer

    def _load_model(self):
        model = LlamaForCausalLM.from_config(self.config)
        return model

    def _configure_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=2e-5)

    def _load_dataset(self):
        data_path = 'alpaca_evol_instruct_70k.json'
        dataset = CollieDatasetForTraining.from_json(data_path, self.tokenizer, max_length=self.args.seqlen)
        # TEST
        train_dataset = dataset[:640]
        return train_dataset

    def train(self):
        monitors = [StepTimeMonitor(self.config), TGSMonitor(self.config), MemoryMonitor(self.config), LossMonitor(self.config)]
        trainer = Trainer(
            model=self.model,
            loss_fn=GPTLMLoss(-100),
            tokenizer=self.tokenizer,
            config=self.config,
            optimizer=self.optimizer,
            train_dataset=self.train_dataset,
            monitors=monitors
        )
        trainer.train()

def init(args):
    import deepspeed.runtime.pipe.module
    parts_partition_arr = list(map(int, args.partitions.split(',')))
    os.environ['parts_partition_arr'] = str(parts_partition_arr)
    deepspeed.runtime.pipe.module.PipelineModule._partition_layers = _partition_layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified partitions and configuration.')
    parser.add_argument('--partitions', type=str, default="0,8,19,29", help='A comma-separated list of numbers representing parts_partition_arr.')
    parser.add_argument('--gpuType', type=str, default="A100", help='GPU type')
    parser.add_argument('--syn', action='store_true', help='GPU synchronize engine.py')
    parser.add_argument('--baseDir', type=str, default="partition_data", help='data dir')
    parser.add_argument('--seqlen', type=int, default=1400, help='sequence length')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--modelpath', type=str, default="llama_3b", help='model path')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='gradient_accumulation_steps')
    parser.add_argument('--dpSize', type=int, default=1, help='data parallel size')
    parser.add_argument('--train_epochs', type=int, default=1, help='training epoches')
    args = parser.parse_args()
    init(args)
    trainer = EspressoTrainer(args)
    trainer.train()
