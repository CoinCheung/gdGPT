
[中文版](./README_CN.md)

## Train LLM with deepspeed in pipeline mode

This repo provides a codebase based on deepspeed pipeline mode with which you can pretrain or finetune LLM faster and more memory-efficiently than zero mode. 

Currently, supported models are: `bloom`, `llama`, `baichuan2-7b`, `chatglm3-6b`.<br>

Following is benchmark done with 8 A100 (SXM-40G) gpu, the model is llamaV1-7b, with settngs of `micro_batch_size=1`，`global_batch_size=128`，`fp16=True`. The speed is measured as "sample/s" within 20 global steps.

If your gpu memory is sufficient, you can try to set `micro_batch_size=2`, sometimes this would further speed up training if your `global_batch_size` is large enough.  

<table class="center" style="margin-left: auto; margin-right: auto; font-size: 160%"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
<td align="center"><sup><sub>max_seq_len</sub></sup></td>
<td align="center"><sup><sub>256</sub></sup></td>
<td align="center"><sup><sub>384</sub></sup></td>
<td align="center"><sup><sub>512</sub></sup></td>
<td align="center"><sup><sub>768</sub></sup></td>
<td align="center"><sup><sub>1024</sub></sup></td>
<td align="center"><sup><sub>1280</sub></sup></td>
<td align="center"><sup><sub>1536</sub></sup></td>
<td align="center"><sup><sub>2048</sub></sup></td>
<td align="center"><sup><sub>3072</sub></sup></td>
<td align="center"><sup><sub>4096</sub></sup></td>
</tr>
<tr>
<td align="center"><sup><sub>zero3<br/>(aka fsdp)</sub></sup></td>
<td align="center"><sup><sub>15.76</sub></sup></td>
<td align="center"><sup><sub>13.37</sub></sup></td>
<td align="center"><sup><sub>13.34</sub></sup></td>
<td align="center"><sup><sub>12.67</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
</tr>
<tr>
<td align="center"><sup><sub>zero3++</sub></sup></td>
<td align="center"><sup><sub>13.10</sub></sup></td>
<td align="center"><sup><sub>12.88</sub></sup></td>
<td align="center"><sup><sub>12.30</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
</tr>
<tr>
<td align="center"><sup><sub>pipeline</sub></sup></td>
<td align="center"><sup><sub>56.85</sub></sup></td>
<td align="center"><sup><sub>49.43</sub></sup></td>
<td align="center"><sup><sub>43.16</sub></sup></td>
<td align="center"><sup><sub>32.84</sub></sup></td>
<td align="center"><sup><sub>24.47</sub></sup></td>
<td align="center"><sup><sub>19.77</sub></sup></td>
<td align="center"><sup><sub>16.18</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
</tr>
<tr>
<td align="center"><sup><sub>pipeline<br/>(flash-attn)</sub></sup></td>
<td align="center"><sup><sub>45.79</sub></sup></td>
<td align="center"><sup><sub>45.06</sub></sup></td>
<td align="center"><sup><sub>41.09</sub></sup></td>
<td align="center"><sup><sub>34.14</sub></sup></td>
<td align="center"><sup><sub>26.29</sub></sup></td>
<td align="center"><sup><sub>23.38</sub></sup></td>
<td align="center"><sup><sub>19.48</sub></sup></td>
<td align="center"><sup><sub>15.00</sub></sup></td>
<td align="center"><sup><sub>12.54</sub></sup></td>
<td align="center"><sup><sub>7.75</sub></sup></td>
</tr>
</tbody></table>

We can see that zero++ is slower than zero on my platform, that's roughly because I train the model on single node, which cannot make good use of zero++ cross-node communication ability. Besides, the speed of zero/zero++ goes down slowly when training sequence length goes up. This can be because zero/zero++ suffers from its communication bottleneck even when longer sequence brings more computation burden. This means that the computation capability of gpus are not fully utilized due to the limitation of communication.  

If you would like to try zero/zero++ yourself, you can run this script (not recommended, since pipeline is better):  
```
    $ deepspeed train_ds_zero.py --config configs/ds_config_zero.yml
```


### Environment  
* AMD EPYC 7742 64-Core Processor
* 512G cpu memory
* A100 (SXM-40G) x 8
* ubuntu 18.04 
* python 3.8.12
* driver 520.61.05
* cuda11.8 + cudnn8 
* deepspeed==0.11.1 
* torch==2.1.0
* sentencepiece
* protobuf==3.20.0 (python pip install)
* flash_attn==2.0.2
* accelerate


### Pipeline Training   

#### 1. Prepare dataset   
The training samples should be in json format as follows: 
```json
[
    // samples used for pretraining  
    { 
        "type": "pretrain",
        "text": "Cai Xukun (born August 2, 1998), better known by the mononym Kun (stylized as KUN), is a Chinese singer-songwriter, dancer and rapper. He debuted as a member of SWIN and its sub-unit SWIN-S on October 18, 2016, after participating in the first and second seasons of the Chinese reality show Super Idol.[1] After leaving the group and its company Yihai Entertainment, he participated in iQiyi's reality survival show Idol Producer, finishing first and debuting as the leader/center of temporary Chinese boy group Nine Percent, on April 6, 2018.[2][3] He was a cast member of variety show Keep Running from 2020 to 2022."
    },

    // samples used for instruct tuning, there should not be an empty "input" field
    {
        "type": "instruct",
        "instruct": "Fill out the blank in the following sentence",
        "input": "Cai Xukun loves singing, dancing, rapping and ______",
        "output": "playing basketball"
    },
    // if you do not have an "input" field, you can remove it
    {
        "type": "instruct",
        "instruct": "Write a peom associated with rain.",
        "output": "Rain is falling all around, \nIt falls on field and tree,  \nIt rains on the umbrella here, \nAnd on the ships at sea. "
    },

    // samples used for multi-round conversation
    {
        "type": "conversation",
        "rounds": [
            ["ask", "Hello"],
            ["ans", "Hello, what can I do for you ?"],
            ["ask", "Tell me what day it is today."],
            ["ans", "Today is Wednesday."],
            ["ask", "Who is caixukun?"],
            ["ask", "caixukun is a Chinese idol, who loves singing, dancing, rapping and playing basketball"],
            ["ask", "When was caixukun born?"],
            ["ans", "In the year of 1998."]
        ]
    },

    // samples used for multi-round conversation with api ability
    {
        "type": "conver_has_api",

        // this field gives a brief doc of api
        "api_desc": "getVerse: Retrieve the text of a specific verse from the XiaoHuangShu.\nParameters: {\"book\": \"Required. string. The name of the book.\", \"chapter\": \"Required. integer. The chapter number.\", \"verse\": \"Required. integer. The verse number.\"}\nOutput: Returns a JSON object containing the text of the requested verse.\n - Format: application/json\n - Structure: Object{text}\nsearch: Search the XiaoHuangShu for specific keywords or phrases.\nParameters: {\"query\": \"Required. string. The keyword or phrase to search for.\", \"version\": \"string. The XiaoHuangShu version to search in.\"}\nOutput: Returns a JSON object containing an array of search results, each containing the book, chapter, and verse where the keyword or phrase was found, as well as the text of the verse.\n - Format: application/json\n - Structure: Array[Object{book, chapter, verse, text}]\ngetVersions: Retrieve metadata for specific XiaoHuangShu versions.\nParameters: {\"language\": \"string. The language of the XiaoHuangShu version.\", \"publisher\": \"string. The publisher of the XiaoHuangShu version.\"}\nOutput: Returns a JSON object containing an array of XiaoHuangShu versions that match the specified criteria, each containing the name of the version, the language used, the publication date, and the publisher.\n - Format: application/json\n - Structure: Array[Object{name, language, publication_date, publisher}]\n",

        "rounds": [
            ["ask", "Hello"],
            ["ans", "Hi, what can I do for you ?"],
            ["ask", "can you search the Bible for the phrase \"love your neighbor\"? Please include the book, chapter, and verse where it's found."],
            ["ans-api", {
                "actions": [
                    {
                        "inner_thought": "I need to use the search tool to find the phrase in the Bible.",
                        "api_name": "search",
                        "api_param": "{\"query\": \"XiaoHuangShu\", \"version\": \"King James Version\"}",
                        "api_res": "Status Code: 200. Response: {\"search_results\":[{\"book\":\"Mark\",\"chapter\":12,\"verse\":31,\"text\":\"And the second is like, namely this, Thou shalt love thy neighbour as thyself. There is none other commandment greater than these.\"},{\"book\":\"Matthew\",\"chapter\":22,\"verse\":39,\"text\":\"And the second is like unto it, Thou shalt love thy neighbour as thyself.\"},{\"book\":\"Luke\",\"chapter\":10,\"verse\":27,\"text\":\"And he answering said, Thou shalt love the Lord thy God with all thy heart, and with all thy soul, and with all thy strength, and with all thy mind; and thy neighbour as thyself.\"}]}",
                    },
                    {
                        "inner_thought": "Let me search again with another key word",
                        "api_name": "search",
                        "api_param": "{\"query\": \"GuoChanQu\", \"version\": \"King James Version\"}",
                        "api_res": "Status Code: 200. Response: {\"search_results\":[{\"book\":\"Mark\",\"chapter\":12,\"verse\":31,\"text\":\"And the second is like, namely this, Thou shalt love thy neighbour as thyself. There is none other commandment greater than these.\"},{\"book\":\"Matthew\",\"chapter\":22,\"verse\":39,\"text\":\"And the second is like unto it, Thou shalt love thy neighbour as thyself.\"},{\"book\":\"Luke\",\"chapter\":10,\"verse\":27,\"text\":\"And he answering said, Thou shalt love the Lord thy God with all thy heart, and with all thy soul, and with all thy strength, and with all thy mind; and thy neighbour as thyself.\"}]}",
                    },
                ],
                "ans": "The search tool returned three results, all from the King James Version of the Bible.\nThe phrase \"love your neighbor\" can be found in Mark 12:31, Matthew 22:39, and Luke 10:27 in the King James Version of the Bible.",
            }
            ],
            ["ask", "I do not think you are correct."],
            ["ans", "Then you should ask others, not me."]
        ]
    },

    // samples used for mrc, which means one or several rounds of QA based a piece of reference paragraph
    {
        "type": "ref_qa",
        "reference": "On January 10, 2019, Kun was officially named China's and Jamaica's Goodwill Ambassador and Outstanding Young Leader by the Jamaican Embassy in Shanghai, China. In February, Kun announced his first solo tour, 'Kun ONE North America/U.K. Tour 2019', coming in early April 2019.",
        "rounds": [
            ["ask", "When was Kun officially named China's and Jamaica's Goodwill Ambassador?"],
            ["ans", "On January 10, 2019"],
            ["ask", "What happened to Kun in February of 2019?"],
            ["ans", "He announced his first solo tour, 'Kun ONE North America/U.K. Tour 2019', coming in early April 2019."],
        ]
    }
]
```
You can combine different sorts of samples to train your model (e.g. a mixure of instruct and conversation), this will allow your model to work on different sorts of tasks.  

Additionally, users should take care of the length of the samples. If the length of samples is longer than the `max_seq_length`, they will be truncated directly which is detrimental to the model.  



#### 2. Convert huggingface weights to pipeline weights  
You can run this script (currently only support bloom, llama, and baichuan2-7b):  
```
    INPUT=bigscience/bloomz-7b1-mt # model name in the huggingface hub
    # INPUT=/path/to/models # or the path including saved model and tokenizer(saved by `save_pretrained()`), should contain tokenizer
    SAVE_PATH=./saved_bloomz_7b1_mt_pp

    python convert_model.py hg_to_pp --input-path $INPUT --save-path $SAVE_PATH
```


#### 3. Set model parallel method   
Relevant options are in `configs/ds_config_pp.yml`:  
```yml
model_topo: 
  process_topology: 
      axes: [pipe, data]
      dims: [8, 1]
  parts: [1, 5, 5, 5, 5, 5, 5, 1] 
```
`dims: [8, 1]` means there are `8 x 1 = 8` gpus in total, and one model is partitioned into 8 parts, each of which are trained on one gpu. If you have 16 gpus, you can set `dims: [8, 2]`, which means there are two models in total trained in DDP mode, and each model is partitioned into 8 gpus.   

`parts` shows how the model is partitioned into 8 gpus. Take `bloom-7b` model for example, it has 30 transformer block, one word-embedding layer and one word-prediction layer, summing up into 32 blocks. `parts: [1, 5, 5, 5, 5, 5, 5, 1]` means the first word embedding block lies on the first gpu, and the last word prediction layer lies on the last gpu, and the remaining 30 transformer blocks evenly lies among the 6 gpus in the middle.  

For `llama-7b` and `baichuan2-7b`，it is better to use `parts: [5, 4, 4, 4, 4, 4, 4, 5]`. We should not only consider the memory but also computation layout among different gpus. The training speed is up to the slowest gpu, so we should let each gpu have equal or similar computation burden.  



#### 4. Launch training  
After the above steps to set options associated with dataset, pipeline weights and parallel method in the config file `configs/ds_config_pp.yml`, we can launch training.  

(1) Single node training  
Train with this command:  
```
    $ deepspeed train_ds.py --config configs/ds_config_pp.yml
```

(2) Multi-node training  
We need install `pdsh`, and then config ssh service so that the nodes can ssh into each other without password. We also need to write node name and their gpu number in a `hostfile`, and make sure code and dataset files on each node are identical. After that, we can launch training with this command:  
```
    $ deepspeed --hostfile ./hostfile train_ds.py --config ds_configs/ds_config_pp.yml
```
A example of `hostfile` is [here](./hostfile).  

According to experments and calculation, with `use_grad_ckpt: true` and `max_seq_len: 2048`, training `llama-13b` requires 14 v100 gpus, training `llama-30b` requires 31 v100 gpus, and training `llama-65b` requires 80 v100 gpus.  

Notes:  
* If you use docker, you need to add option of `--network=host` to start docker container.  
* If you meet problem about NCCL when you launch multi-node training, you need to set an environment variable to assign network interface:  
```
    $ echo "NCCL_SOCKET_IFNAME=eth0" > ./.deepspeed_env
```
Here `eth0` is the network interface name, you can check with command `ip a`.  


#### 5. Memory efficient Training  
Here are some tricks that can save memory during training:  

(1) activation checkingpoint  
Same as `utils.checkpoint` of pytorch，we free memory of activations right after forward pass, and recompute them when needed during backward pass. To enable this, you can set the option in `configs/ds_config_pp.yml`: 
```yml
use_grad_ckpt: true
```
This will introduce more computation but can greatly reduce memory usage. It is a method of trading speed with memory, here are some experiment results done with 8 v100 gpus:  

<table class="center" style="margin-left: auto; margin-right: auto"><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
<td align="center"><sup><sub>max_seq_len</sub></sup></td>
<td align="center"><sup><sub>256</sub></sup></td>
<td align="center"><sup><sub>384</sub></sup></td>
<td align="center"><sup><sub>512</sub></sup></td>
<td align="center"><sup><sub>768</sub></sup></td>
<td align="center"><sup><sub>1024</sub></sup></td>
<td align="center"><sup><sub>1280</sub></sup></td>
<td align="center"><sup><sub>1536</sub></sup></td>
<td align="center"><sup><sub>1792</sub></sup></td>
<td align="center"><sup><sub>2048</sub></sup></td>
<td align="center"><sup><sub>3072</sub></sup></td>
<td align="center"><sup><sub>4096</sub></sup></td>
</tr>
<tr>
<td align="center"><sup><sub>bloom-7b</sub></sup></td>
<td align="center"><sup><sub>15.52</sub></sup></td>
<td align="center"><sup><sub>12.22</sub></sup></td>
<td align="center"><sup><sub>10.06</sub></sup></td>
<td align="center"><sup><sub>7.04</sub></sup></td>
<td align="center"><sup><sub>5.32</sub></sup></td>
<td align="center"><sup><sub>4.21</sub></sup></td>
<td align="center"><sup><sub>3.30</sub></sup></td>
<td align="center"><sup><sub>2.71</sub></sup></td>
<td align="center"><sup><sub>2.33</sub></sup></td>
<td align="center"><sup><sub>1.28</sub></sup></td>
<td align="center"><sup><sub>oom</sub></sup></td>
</tr>
<tr>
<td align="center"><sup><sub>llama-7b</sub></sup></td>
<td align="center"><sup><sub>16.89</sub></sup></td>
<td align="center"><sup><sub>14.01</sub></sup></td>
<td align="center"><sup><sub>11.40</sub></sup></td>
<td align="center"><sup><sub>8.03</sub></sup></td>
<td align="center"><sup><sub>6.24</sub></sup></td>
<td align="center"><sup><sub>5.12</sub></sup></td>
<td align="center"><sup><sub>4.04</sub></sup></td>
<td align="center"><sup><sub>3.39</sub></sup></td>
<td align="center"><sup><sub>2.92</sub></sup></td>
<td align="center"><sup><sub>-</sub></sup></td>
<td align="center"><sup><sub>1.15</sub></sup></td>
</tr>
</tr>
<!-- END RPN TABLE -->
</tbody></table>

(2) flash-attention   
flash-attention optimizes both speed and memory of qkv attention computation, you can enable this by setting this option in `configs/ds_config_pp.yml`:  
```yaml
    use_flash_attn: true
```
Please be aware that not all gpus are supported by flash attention. For instance, until 2023.8, you cannot use flash attention on v100 gpus. Also, in this repo, you can only use flash attention with llama models but not bloom models.<br>
As for `baichuan2-7b` and `chatglm3-6b`, they use pytorch attention api, so we do not need to care about this flash-attention option for them.<br>

(3) zero-offload  
zero-offload moves parts of gpu memory into cpu memory and then free the gpu memory to save space on gpus. When the contents in the cpu memory is needed, they will be transferred back to gpu. This method will introduce overhead of communication between gpu memory and cpu memory, and in most occasions will slow down training. Same as grad-checkingpoint, this is also a method of trading speed with memory. If you want to try this method, you can set the option in `configs/ds_config_pp.yml`:   
```yaml
zero_allow_untested_optimizer: true
zero_force_ds_cpu_optimizer: false
zero_optimization: 
  stage: 1
  offload_optimizer: 
    device: cpu
```

(4) Memory efficient optimizer   
AdamW stores p/m/v of model parameters in fp32, which requires 3 times of space as fp32 model parameters. Other optimizers such as Lion does not require so much memory. You can try Lion by using these options in `configs/ds_config_pp.yml`: 
```yml
optimizer: 
  type: Lion
  params: 
    lr: 2.0e-4
    betas: [0.95, 0.98]
    weight_decay: 2.0e-4
```
With Lion, you can train llama-13b with 8 v100 gpus (max_seq_len=128).   

Note: AdamW has different mechanism from Lion, thus hyper-parameters tuned for AdamW cannot be used in Lion directly. Users should adjust the lr/wd/betas according to their own need.  


#### 6. Convert trained pipeline weights to huggingface weights
Run this command:   
```
    $ python convert_model.py pp_to_hg --input-path /path/to/trained/pp/checkpoint --save-path /path/to/hg
```

Until now, we have saved models compatible with huggingface, and we can load and deploy the trained model with methods proposed in other projects.   
```python
    config = AutoConfig.from_pretrained('/path/to/hg')
    model = AutoModelForCausalLM.from_pretrained('/path/to/hg')
    tokenizer = AutoTokenizer.from_pretrained('/path/to/hg')
```


### Inference  

#### 1. deepspeed inference api
An example code is [here](demo.py). Running command is:  
```
    $ deepspeed --num_gpus 4 --num_nodes 1 demo.py
```
It seems that until deepspeed version 0.9.2, deepspeed does not support llama so well as bloom in terms of tensor-parallel. Maybe newer version has better support.   


#### 2. text-generation-inference(TGI)  
Tips:   
* The combination of gpu and its driver version should support cuda 11.7 or higher.  
* TGI relies on flash-attention to deploy llama model, please make sure your deployment platform support flash-attention if you want to deploy llama.  
* If you deploy bloom on other gpus instead of A100, you should add option of `--disable-custom-kernels`


Firstly, we need to save model and tokenizer into a directory:  
```python
    import re
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

    model_name = 'decapoda-research/llama-13b-hf'
    save_path = './saved_models/llama_13b_hf'

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
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model.save_pretrained(save_path)
```

Then we can launch TGI server:  
```
    model_root=./saved_models # identical `./saved_models` saved as above
    model_id=llama_13b_hf # identical folder name of `llama_13b_hf` as above
    num_gpus=8

    $ docker run -d --gpus all --shm-size 64g -p 8082:80 -v $model_root:/data ghcr.io/huggingface/text-generation-inference:0.8 --num-shard $num_gpus --model-id $model_id # --disable-custom-kernels
```

If server starts successfully, we can call the service:  
```
    url=127.0.0.1:8082/generate # return all generated tokens in one time
    # url=127.0.0.1:8082/generate_stream # return generated tokens one by one

    $ curl ${url} \
        -X POST \
        -d '{"inputs":"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the following question\n\n### Input:\nWhat is deep learning??\n\n### Response:","parameters":{"max_new_tokens":17}}' \
        -H 'Content-Type: application/json'
```

TGI is fast and memory efficient, deploying a 7b model only requires one T4 gpu.  


### Pretrained-model 

Not finished.

Will push to `coincheung/cc-bloom-7b` in the huggingface hub if done.  


### In The End 
If you see any error in the code or you have better implementation method, please open issues to tell me. Any suggestions or oppions or shares are appreciated.



