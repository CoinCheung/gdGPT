
import os
import re

import torch
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import pipeline

torch.set_grad_enabled(False)


def infer_with_deepspeed(model_name, txt):
    '''
        deepspeed的方式，每个gpu上有一个进程，每个进程都加载一遍完整的模型，容易导致oom
        运行:
            deepspeed --num_gpus 4 --num_nodes 1 demo.py
    '''
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    config = AutoConfig.from_pretrained(model_name)
    if re.search('llama', model_name):
        tokenizer = LlamaTokenizer.from_pretrained(model_name, config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)

    infer_config = dict(
            tensor_parallel={'tp_size': world_size},
            dtype=torch.half,
            replace_with_kernel_inject=True,
    )
    ## 使用pipeline
    model = pipeline('text-generation', model=model_name,
            device=local_rank,
            torch_dtype=torch.half,
            tokenizer=tokenizer,
    )
    model.model = deepspeed.init_inference(model.model, config=infer_config)
    res = model([txt,], do_sample=False, temperature=0.7, max_new_tokens=300)

    return res


if __name__ == '__main__':

    prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    简化以下段落，使其更易理解

    ### Input:
    尽管人们普遍认为互联网使我们能够与世界各地的人联系，但仍有一些人不熟悉其基本功能，不理解为什么它变得如此普遍，或者它的真正能力是什么。

    ### Response:'''

    model_name = 'decapoda-research/llama-7b-hf'
    #  model_name = 'bigscience/bloomz-560m'

    res = infer_with_deepspeed(model_name, prompt)
    print(res)

