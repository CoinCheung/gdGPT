
'''
首先把模型都保存到一个目录:
```python
    import re
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

    model_name = 'decapoda-research/llama-30b-hf'
    save_path = '/data/zzy/models/llama_30b_hf'

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    if re.search('^bigscience/bloom', model_name):
        model.lm_head.weight = nn.Parameter(
            model.transformer.word_embeddings.weight.clone())
    if re.search('^decapoda-research/llama', model_name):
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
```

启动服务，并且把模型挂载进去:
```
    image=ghcr.io/huggingface/text-generation-inference:0.7 # 这个0.7不支持v100，改成0.6
    model_root=/data/zzy/models
    model_id=/data/llama_30b_hf # 里面的from_pretrained的位置
    num_shard=8 # 分成几张卡
    port=8080 # 从外面调用时候使用的端口

    docker run -d --gpus all --rm --shm-size 64g -p $port:80 -v $model_root:/data \
            $image \
            --num-shard $num_shard \
            --model-id $model_id
```
'''


import json
import requests

'''
curl -N [YOUR_IP]:[YOUR_PORT]/generate_stream -X POST -d '{"inputs":"Below is...\n\n### Instruction\n天空为什么是蓝色的\n\n### Response\n","parameters":{"max_new_tokens":256, "stop":["</s>"]}}' -H 'Content-Type: application/json'
'''

url = 'http://10.128.61.27:8080/generate'
#  url = 'http://10.128.61.28:8080/generate_stream'

def call_service_request(instruct):

    inp = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruct}\n\n### Response:\n'
    #  inp = instruct

    headers = {'Content-Type': 'application/json'}
    data = {
        "inputs": inp,
        "parameters": {
            "max_new_tokens": 512,
            "stop": ["</s>",]
        },
    }

    ret = requests.post(url, json=data, headers=headers)
    print(ret.text)
    res = json.loads(ret.text)['generated_text']
    return res


def call_service_client(txt):
    from text_generation import Client

    client = Client("http://10.128.61.28:8080")
    print(client.generate(txt, max_new_tokens=512).generated_text)

    text = ""
    for response in client.generate_stream("What is Deep Learning?", max_new_tokens=17):
        if not response.token.special:
            text += response.token.text
    print(text)


func = call_service_request
#  func = call_service_client
#res = func('天空为什么是蓝色的')
res = func('What is DeepLearning')
print(res)
