
import os
import re
import torch
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import pipeline
import time

torch.set_grad_enabled(False)



##=================================
## 使用transformers自带的pipeline
##=================================
def infer_with_transformers_pipeline(model_name, txt):

    config = AutoConfig.from_pretrained(model_name)
    if re.search('llama', model_name):
        tokenizer = LlamaTokenizer.from_pretrained(model_name, config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)

    model = pipeline('text-generation', model=model_name,
            torch_dtype=torch.half,
            tokenizer=tokenizer,
            #  device=local_rank,
            device_map='auto')
    res = model([txt,], do_sample=False, temperature=0.7, max_new_tokens=300)
    print(res)


##=================================
## 使用deepspeed的inference api
##=================================
def infer_with_deepspeed(model_name, txt):
    '''
        ddp的方式，每个进程都加载一遍模型，会导致oom
        好像不太对劲，使用deepspeed --num_gpus 4 --num_nodes 1 demo.py时，并没有tp运行，而是每个gpu都运行了一遍，并且返回了4个结果，然后显存占用也是一点没减小
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
            #  device_map='auto',
    )
    model.model = deepspeed.init_inference(model.model, config=infer_config)
    res = model([txt,], do_sample=False, temperature=0.7, max_new_tokens=300)

    print(res)


if __name__ == '__main__':

    prompt = '''Summarize this for a second-grade student:

    Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.'''

    model_name = 'decapoda-research/llama-7b-hf'
    model_name = 'bigscience/bloomz-560m'
    #  infer_with_transformers_pipeline(model_name, prompt)
    infer_with_deepspeed(model_name, prompt)

