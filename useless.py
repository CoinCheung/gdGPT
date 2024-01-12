
import os
import re

import torch
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import pipeline

torch.set_grad_enabled(False)

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

def create_tokenizer(model_name):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    if re.search('llama', model_name):
        tokenizer = LlamaTokenizer.from_pretrained(model_name, config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, trust_remote_code=True)

    return tokenizer

model_name = 'checkpoints/tool_alpaca'
tokenizer = create_tokenizer(model_name)


#  print(tokenizer.__dir__())
print(tokenizer.special_tokens)
print(tokenizer.eos_token)
print(tokenizer.eos_token_id)
print(tokenizer(tokenizer.eos_token, add_special_tokens=False).input_ids)
print([tokenizer.decode(el) for el in tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)])
print([tokenizer.decode(el) for el in tokenizer.encode(' </s>', add_special_tokens=False)])

print([tokenizer.decode(el) for el in [1833, 30917, 30994]])

model_name = 'checkpoints/tool_alpaca'
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, trust_remote_code=True)
print([tokenizer.decode(el) for el in [1833, 2893]])
