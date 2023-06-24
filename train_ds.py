import json
import os
import re
import random
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import BloomConfig, BloomForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from datasets import *
from models import *

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe import ProcessTopology
from deepspeed.runtime.config import DEEPSPEED_OPTIMIZERS
from deepspeed.compression.compress import init_compression, redundancy_clean




max_seq_len = 640
#  max_seq_len = 512
#  max_seq_len = 1024
#  max_seq_len = 2048
n_epoches = 3
n_save_iters = 1000
save_path = '/nfs-data/zzy/checkpoints/'
use_grad_ckpt = False
use_qat = False

## bloom
#  model_name = 'bigscience/mt0-xxl'
#  model_name = 'bigscience/bloomz-7b1-mt'
#  model_name = 'bigscience/bloomz-560m'
#  model_name = 'bigscience/bloom-7b1'
#  model_name = 'coincheung/bloomz-7b1-mt-llt'
#  model_name = 'coincheung/bloomz-7b1-mt-nvl-cllv'
#  model_name = 'coincheung/bloomz-7b1-mt-org-prune'
#  load_path = '/data/zzy/checkpoints/model_bloomz_7b1_mt_org'
#  load_path = '/data/zzy/checkpoints/model_bloom_7b1_org'
#  load_path = '/nfs-data/zzy/checkpoints/model_bloom_7b1_200w_lion'
#  load_path = '/nfs-data/zzy/checkpoints/model_novel_ali2_final'
#  load_path = '/nfs-data/zzy/checkpoints/bloomz-7b1-mt-nvl-cllv'
#  load_path = '/nfs-data/zzy/checkpoints/model_bloom_7b1_from_ali2novel_mqa_novel_80w_l640_coslr'
#  load_path = '/nfs-data/zzy/checkpoints/checkpoint_final'
#  load_path = None
#  num_pp_stages = 8

## lama
#  model_name = 'decapoda-research/llama-7b-hf'
#  load_path = '/data/zzy/checkpoints/model_llama_7b_hf_org'
#  #  load_path = None
#  num_pp_stages = 8
#  topo = ProcessTopology(axes=['data', 'pipe'], dims=[1, 8])

#  parts = [5, 4, 4, 4, 4, 4, 4, 5] # best for llama-7b
#  parts = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] # best for llama-13b x 14 gpu
#  parts = [4 for _ in range(8)] # coincheung/bloomz-7b
#  parts = [1, 5, 5, 5, 5, 5, 5, 1] # bigscience/bloomz-7b 8gpu

model_path = '/nfs-data/zzy/checkpoints/bloomz-7b1-mt-novel-ali2-final_pp'
from_scratch = False
parts = [1, 5, 5, 5, 5, 5, 5, 1]
topo = ProcessTopology(axes=['pipe', 'data'], dims=[8, 1])


## dataset
#  dataset_cls = TextDataSetShards
#  data_path = '/nfs-data/zzy/novels/novel_files'
dataset_cls = TextDataSet
#  data_path = './datasets/novel_512_200w.txt'
#  data_path = './datasets/novel_write_230.txt'
#  data_path = './datasets/novel_512_40w_instruct_gen_train.json'
#  data_path = './datasets/processed_multiturn_chat_0.8M.json.train'
#  data_path = './datasets/processed_RefGPT-Dataset-V1-CN.json.train'
data_path = '../datasets/novel_and_multiturn.json'



parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--config', type=str)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

device = torch.device('cuda', args.local_rank)
torch.cuda.set_device(args.local_rank)

ws = int(os.environ['WORLD_SIZE'])
rk = int(os.environ['RANK'])


#  with open(args.deepspeed_config, 'r') as fr:
with open(args.config, 'r') as fr:
    ds_cfg = json.load(fr)


def get_model(model_path, grad_ckpt=False):
    config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type

    kwargs = {'config': config, 'load_path': model_path, 'grad_ckpt': grad_ckpt, }
    if from_scratch: kwargs['load_path'] = None
    if hasattr(config, 'tie_word_embeddings'):
        kwargs['tie_emb'] = config.tie_word_embeddings

    if re.search('llama', model_type):
        specs = get_llama_causal_lm_specs(**kwargs)
    elif re.search('bloom', model_type):
        specs = get_bloom_causal_lm_specs(**kwargs)

    model = PipelineModule(layers=specs,
                        loss_fn=LMCrossEntropyLoss(),
                        #  num_stages=num_pp_stages,
                        topology=topo,
                        #  partition_method='parameters',
                        #partition_method='uniform',
                        partition_method=parts,
                        activation_checkpoint_interval=0)
    model.hg_config = config
    return model


training_data = dataset_cls(data_path, model_path, max_seq_len)

# global-step数，这个是acc_step=1的情况，如果acc_step=4，
# 就是一个iter取4个batch，这样就是4 x n_iters个batch了，
# 也就是for循环的一个iter里面实际上要在engine里面迭代acc_step次
n_iters = n_epoches * len(training_data) // ds_cfg['train_batch_size']

print('len(training_data): ', len(training_data))
print('n_iters: ', n_iters)

deepspeed.init_distributed(dist_backend='nccl')

model = get_model(model_path, use_grad_ckpt)

# qat settings
if use_qat: ds_cfg['compression_training']['activation_quantization']['shared_parameters']['enabled'] = True
if use_qat: ds_cfg['compression_training']['weight_quantization']['shared_parameters']['enabled'] = True
if use_qat: model = init_compression(model, args.config)

ds_cfg['scheduler']['params']['total_num_steps'] = n_iters
#  ds_cfg['scheduler']['params']['warmup_max_lr'] = ds_cfg['optimizer']['params']['lr']


model_engine, optimizer, train_loader, lr_schdlr = deepspeed.initialize(
        args=args, model=model,
        #model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=training_data,
        config=ds_cfg
        )

# 这个没什么用，因为目前还没有比较好看的恢复dataloader的方法，只能加载权重啥的，数据还是重头来的
#  model_engine.load_checkpoint(save_path, tag='last')


for i in range(n_iters):
    loss = model_engine.train_batch()

    #  if (i + 1) % n_save_iters == 0:
    #      model_engine.save_checkpoint(save_path, client_state={'iter': i},
    #              tag='checkpoint_last')

if use_qat: model = redundancy_clean(model, args.config)


model_engine.save_checkpoint(save_path, tag='checkpoint_final')
if rk == 0:
    training_data.save_tokenizer(f'{save_path}/checkpoint_final')
    model.hg_config.save_pretrained(f'{save_path}/checkpoint_final')
