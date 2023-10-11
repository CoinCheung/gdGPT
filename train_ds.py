import os
import re
import json
import yaml
import random
import argparse

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import *
from models import *

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe import ProcessTopology



parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--config', type=str)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

device = torch.device('cuda', args.local_rank)
torch.cuda.set_device(args.local_rank)

ws = int(os.environ['WORLD_SIZE'])
rk = int(os.environ['RANK'])


with open(args.config, 'r') as fr:
    ds_cfg = yaml.load(fr, Loader=yaml.FullLoader)


def get_model(model_path, grad_ckpt=False):
    config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type

    kwargs = {'config': config, 'load_path': model_path, 'grad_ckpt': grad_ckpt, }
    if ds_cfg['from_scratch']: kwargs['load_path'] = None
    if hasattr(config, 'tie_word_embeddings'):
        kwargs['tie_emb'] = config.tie_word_embeddings
    kwargs['use_flash_attn'] = ds_cfg.get('use_flash_attn', False)

    if re.search('llama', model_type):
        specs = get_llama_causal_lm_specs(**kwargs)
    elif re.search('bloom', model_type):
        specs = get_bloom_causal_lm_specs(**kwargs)

    topo = ProcessTopology(**ds_cfg['model_topo']['process_topology'])
    model = PipelineModule(layers=specs,
                        loss_fn=LMLoss(ds_cfg),
                        topology=topo,
                        partition_method=ds_cfg['model_topo']['parts'],
                        activation_checkpoint_interval=0)
    model.hg_config = config
    return model


training_data = TextDataSet(ds_cfg['data_path'],
                            ds_cfg['model_path'],
                            ds_cfg['max_seq_len'])
n_iters = ds_cfg['n_epoches'] * len(training_data) // ds_cfg['train_batch_size']
ds_cfg['scheduler']['params']['total_num_steps'] = n_iters

deepspeed.init_distributed(dist_backend='nccl')

model = get_model(ds_cfg['model_path'], ds_cfg['use_grad_ckpt'])
model_engine, optimizer, train_loader, lr_schdlr = deepspeed.initialize(
        args=args, model=model,
        training_data=training_data,
        config=ds_cfg
        )

# 这个没什么用，因为目前还没想到比较好看的恢复dataloader的实现，这个只能加载权重啥的，数据还是重头来了
#  model_engine.load_checkpoint(save_path, tag='last')


save_path = ds_cfg['save_path']
for i in range(n_iters):
    loss = model_engine.train_batch()

    #  n_save_iters = 1000
    #  if (i + 1) % n_save_iters == 0:
    #      model_engine.save_checkpoint(save_path, client_state={'iter': i},
    #              tag='checkpoint_last')



model_engine.save_checkpoint(save_path, tag='checkpoint_final')
if rk == 0:
    training_data.save_tokenizer(f'{save_path}/checkpoint_final')
    model.hg_config.save_pretrained(f'{save_path}/checkpoint_final')
