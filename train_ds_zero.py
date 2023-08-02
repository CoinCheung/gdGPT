import json
import os
import random
import yaml
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import BloomConfig, BloomForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import torch
import deepspeed

from datasets import *



parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--config', type=str)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

device = torch.device('cuda', args.local_rank)
torch.cuda.set_device(args.local_rank)

ws = int(os.environ['WORLD_SIZE'])
rk = int(os.environ['RANK'])

deepspeed.init_distributed(dist_backend='nccl')

with open(args.config, 'r') as fr:
    ds_cfg = yaml.load(fr, Loader=yaml.FullLoader)

model_name = ds_cfg['model_path']
if ds_cfg['from_scratch']:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
if ds_cfg['use_grad_ckpt']: model.gradient_checkpointing_enable()
model.train()
model.to(device)

training_data = TextDataSet(ds_cfg['data_path'],
                            ds_cfg['model_path'],
                            ds_cfg['max_seq_len'])

n_iters = ds_cfg['n_epoches'] * len(training_data) // ds_cfg['train_batch_size']
ds_cfg['scheduler']['params']['total_num_steps'] = n_iters
model_engine, optimizer, train_loader, lr_schdlr = deepspeed.initialize(
        args=args, model=model,
        #  model_parameters=model.parameters(),
        training_data=training_data,
        config=ds_cfg,
        )

print('num of samples: ', len(training_data))
print('num of iters: ', n_iters)

save_path = ds_cfg['save_path']
for e in range(ds_cfg['n_epoches']):
    train_loader.data_sampler.set_epoch(e)
    for i, batch in enumerate(train_loader):
        batch = [el.cuda() for el in batch]
        outputs = model_engine(input_ids=batch[0][..., 0],
                attention_mask=batch[0][..., 1], labels=batch[1])
        model_engine.backward(outputs.loss)
        model_engine.step()

    model_engine.save_checkpoint(save_path, client_state={'epoch': e})
