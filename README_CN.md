
[English](./README.md)

## 使用deepspeed的pipeline方式对LLM进行finetune

这个项目没有什么理论上的创新，没有提出茴香豆的新写法，也没发明什么新工具，仅仅是基于现有的方法和库提供一套简洁易扩展的代码，可以在8张v100服务器上训练7b的模型(对全部模型参数做full-finetune的那种训练)，可以在更多gpu上训练更大的模型，也可以联机训练，速度比zero3方法更快，并且支持更长的输入序列长度。    

目前支持的模型有: `bloom`, `llama`, `baichuan2-7b`, `chatglm3-6b`，`mixtral-8x7b`。<br>

下面是在我的8张40G的A100-SXM上测出来的训练速度，使用的模型是llama-7b，设置是`micro_batch_size=1`，`global_batch_size=128`，`fp16=True`，训练20个step看log显示的速度(sample/s)。  

如果gpu内存足够大，并且`global_batch_size`设的也比较大的话，可以考虑增加`micro_batch_size` (比如设为2)，有时候可以进一步加快训练速度。  

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

不知道为啥，我测的zero++的速度是比zero慢的，可能是因为我这是单机训练，不涉及多机之间的通信，所以没有发挥出zero++的优势吧。另外还可以看到，随着输入序列长度的增加，zero的速度减小的比较慢，这说明相对于计算来说模型参数和优化器状态的通信是更大的瓶颈，即使增加一点计算量也不会出现因为计算负荷过大导致的速度下降。  
我把zero的测试代码也放出来了，欢迎南来北往的老爷们批评指正。  
zero的运行命令就是: 
```
    $ deepspeed train_ds_zero.py --config configs/ds_config_zero.yml
```


### 我的环境  
* AMD EPYC 7742 64-Core Processor
* 512G cpu memory
* A100(SXM-40G) x 8
* ubuntu 18.04 
* python 3.8.12
* driver 520.61.05
* cuda11.8 + cudnn8 
* deepspeed==0.11.1 
* torch==2.1.0
* sentencepiece
* transformers==4.36.2
* protobuf==3.20.0 (python pip install)
* accelerate


### 训练  

#### 1. 准备数据  
用下面的格式准备json格式的文件: 
```json
[
    // 做预训练的数据格式
    { 
        "type": "pretrain",
        "text": "我只想说懂得都懂，不懂的我也不多解释，毕竟自己知道就好，细细品吧。你们也别来问我怎么了，利益牵扯太大，说了对你我都没好处，当不知道就行了，其余的我只能说这里面水很深，牵扯到很多东西。详细情况你们自己是很难找的，网上大部分已经删除干净了，所以我只能说懂得都懂。懂的人已经基本都获利上岸什么的了，不懂的人永远不懂，关键懂的人都是自己悟的，你也不知道谁是懂的人也没法请教，大家都藏着掖着生怕别人知道自己懂事，懂了就能收割不懂的，你甚至都不知道自己不懂。只是在有些时候，某些人对某些事情不懂装懂，还以为别人不懂。"
    },

    // instruct tuning的数据格式，如果没有input就直接不加，不要用空字符串啥的
    {
        "type": "instruct",
        "instruct": "补充下面横线上的内容",
        "input": "再多看一眼就会爆炸，________",
        "output": "再。。。再靠近点快被融化?"
    },
    {
        "type": "instruct",
        "instruct": "写一篇拍老板马屁的文章，题目是《xx的十宗罪》。要求以批评的语气来写，看起来像是在批评其实说的都是剥削的还不够狠之类的，比如老板的缺点就是工作太辛苦对下面的人太仁慈了啥的，让老板看完眼前一亮然后发到公司内部的员工论坛上，之后各大媒体争相报道，连公司外面的人都跟着高潮了。",
        "output": "你要是没事干去村头把粪挑了"
    },

    // 多轮对话的数据格式
    {
        "type": "conversation",
        "rounds": [
            ["ask", "你好"],
            ["ans", "你好"],
            ["ask", "今天星期几"],
            ["ans", "今天星期三"],
            ["ask", "明天星期几"],
            ["ask", "昨天星期几"],
            ["ask", "前天星期几"],
            ["ans", "傻逼，再问打死你"]
        ]
    },

    // 调用api，给定api描述，让模型在回答时候可以使用api
    {
        "type": "conver_has_api",

        // 这个字段是一段文档，详细描述这个api是怎么用的
        "api_desc": "getVerse: Retrieve the text of a specific verse from the XiaoHuangShu.\nParameters: {\"book\": \"Required. string. The name of the book.\", \"chapter\": \"Required. integer. The chapter number.\", \"verse\": \"Required. integer. The verse number.\"}\nOutput: Returns a JSON object containing the text of the requested verse.\n - Format: application/json\n - Structure: Object{text}\nsearch: Search the XiaoHuangShu for specific keywords or phrases.\nParameters: {\"query\": \"Required. string. The keyword or phrase to search for.\", \"version\": \"string. The XiaoHuangShu version to search in.\"}\nOutput: Returns a JSON object containing an array of search results, each containing the book, chapter, and verse where the keyword or phrase was found, as well as the text of the verse.\n - Format: application/json\n - Structure: Array[Object{book, chapter, verse, text}]\ngetVersions: Retrieve metadata for specific XiaoHuangShu versions.\nParameters: {\"language\": \"string. The language of the XiaoHuangShu version.\", \"publisher\": \"string. The publisher of the XiaoHuangShu version.\"}\nOutput: Returns a JSON object containing an array of XiaoHuangShu versions that match the specified criteria, each containing the name of the version, the language used, the publication date, and the publisher.\n - Format: application/json\n - Structure: Array[Object{name, language, publication_date, publisher}]\n",

        "rounds": [
            ["ask", "你好"],
            ["ans", "你好，找我干啥"],
            ["ask", "有没有法语写的小黄书呢"],
            ["ans-api", {
                "actions": [
                    {
                        "inner_thought": "擦，我哪懂这个啊，那就调用搜索api试试吧，关键词就用XiaoHuangShu看看行不行",
                        "api_name": "search",
                        "api_param": "{\"query\": \"XiaoHuangShu\", \"version\": \"King James Version\"}",
                        "api_res": "Status Code: 200. Response: {\"search_results\":[{\"book\":\"Mark\",\"chapter\":12,\"verse\":31,\"text\":\"And the second is like, namely this, Thou shalt love thy neighbour as thyself. There is none other commandment greater than these.\"},{\"book\":\"Matthew\",\"chapter\":22,\"verse\":39,\"text\":\"And the second is like unto it, Thou shalt love thy neighbour as thyself.\"},{\"book\":\"Luke\",\"chapter\":10,\"verse\":27,\"text\":\"And he answering said, Thou shalt love the Lord thy God with all thy heart, and with all thy soul, and with all thy strength, and with all thy mind; and thy neighbour as thyself.\"}]}",
                    },
                    {
                        "inner_thought": "结果不是很理想啊，那再试试用关键词GuoChanQu搜搜看",
                        "api_name": "search",
                        "api_param": "{\"query\": \"GuoChanQu\", \"version\": \"King James Version\"}",
                        "api_res": "Status Code: 200. Response: {\"search_results\":[{\"book\":\"Mark\",\"chapter\":12,\"verse\":31,\"text\":\"And the second is like, namely this, Thou shalt love thy neighbour as thyself. There is none other commandment greater than these.\"},{\"book\":\"Matthew\",\"chapter\":22,\"verse\":39,\"text\":\"And the second is like unto it, Thou shalt love thy neighbour as thyself.\"},{\"book\":\"Luke\",\"chapter\":10,\"verse\":27,\"text\":\"And he answering said, Thou shalt love the Lord thy God with all thy heart, and with all thy soul, and with all thy strength, and with all thy mind; and thy neighbour as thyself.\"}]}",
                    },
                ],
                "ans": "我搜了一下没找到相关内容，但是我可以给你随便编点东西出来作为回答: The phrase \"love your neighbor\" can be found in Mark 12:31, Matthew 22:39, and Luke 10:27 in the King James Version of the XiaoHuangShu.\n\n您对我的回答满意吗?",
            } // ans-api
            ],
            ["ask", "不满意"],
            ["ans", "那你问别人去啊"]
        ] // rounds
    },

    // 给一段文本，然后针对文本问答的数据格式
    {
        "type": "ref_qa",
        "reference": "一掐脖子就翻白眼，一松手就吹牛逼，早有布局遥遥领先，拳打谷歌脚踢微软，千秋万代一统江湖",
        "rounds": [
            ["ask", "这段话有几个字"],
            ["ans", "100个字"],
            ["ask", "多少汉字多少英文"],
            ["ans", "你不会自己看?"],
        ]
    }
]
```
友情提示，可以把不同形式的数据合并到一起来训练，比如instruct+conversation这种，可以让模型有能力处理不同形式的任务。  
另外，这里需要用户自己控制数据的长度，代码里面仅仅是按设定的最大句子长度做了一下truncation和padding，对于超长的数据就直接把后面的部分截掉了，如果数据集中有许多超长的数据，可能会影响到模型的效果。



#### 2. 转化模型权重  
把huggingface的pretrain权重转成pipeline的模型权重，运行这个脚本(目前仅支持bloom和llama): 
```
    INPUT=bigscience/bloomz-7b1-mt # huggingface上的模型名称
    # INPUT=/path/to/models # 使用save_pretrained保存的模型和tokenizer，一定要包括tokenizer相关文件
    SAVE_PATH=./saved_bloomz_7b1_mt_pp

    python convert_model.py hg_to_pp --input-path $INPUT --save-path $SAVE_PATH
```


#### 3. 设置模型的pipeline方法  
在`configs/ds_config_pp.yml`里面有这样的配置选项:  
```yml
model_topo: 
  process_topology: 
      axes: [pipe, data]
      dims: [8, 1]
  parts: [1, 5, 5, 5, 5, 5, 5, 1] 
```
这个表示一共有`8x1=8`张gpu，并且8张gpu上只有一个模型，如果是`dims: [8,2]`的话，就表示一共有`8x2=16`张gpu，并且每8张gpu上有一个模型，16张gpu上共有两个模型。  
另外就是`parts`表示一个模型在8张gpu上是怎么分配的，`bloom-7b`的模型共有30个transformer的block，加上两端的embedding共有32个block，`parts: [1, 5, 5, 5, 5, 5, 5, 1]`表示第一张和最后一张gpu上各有1个block(按顺序应该是embedding)，中间的6张gpu上每张有5个block(transformer的block)。  

对于llama-7b模型，建议使用`parts: [5, 4, 4, 4, 4, 4, 4, 5]`。  

友情提示: block的分布方式除了要考虑gpu内存之外，还得考虑每张卡的计算负载，因为训练速度决定于最慢的那张gpu，所以要尽量避免某一个gpu计算量比其他gpu大很多的情况。  



#### 4. 训练  
把上面得到的数据集还有模型文件在`configs/ds_config_pp.yml`里面配置好，然后执行训练脚本:  

(1) 单机训练  
可以运行这个命令:  
```
    $ deepspeed train_ds.py --config configs/ds_config_pp.yml
```

(2) 多机训练  
当8张v100不太够用的时候，就得用多机联机训练。首先需要安装pdsh，然后配置一下ssh服务让不同结点之间可以使用ssh免密登陆，再根据ssh结点名配置编辑hostfile，用下面的命令来启动，这个过程需要保证每台服务器上的代码和各种文件**完全相同**:  
```
    $ deepspeed --hostfile ./hostfile train_ds.py --config ds_configs/ds_config_pp.yml
```
hostfile的格式可以参考这个示例的[hostfile](./hostfile)文件。  

经过实验和推算，当打开`gradient checkpointing`并且将`max_seq_len`设为2048时，使用AdamW优化器，训练llama-13b模型需要14张v100，训练llama-30b需要31张v100，训练llama-65b需要80张v100。  

请注意:  
* 如果你在docker环境做多机训练的话，需要在启动docker时加上`--network=host`选项。  
* 如果在多机并行的时候遇到NCCL的问题，需要加上一个环境变量用来指定网卡名:  
```
    $ echo "NCCL_SOCKET_IFNAME=eth0" > ./.deepspeed_env
```
这里面的`eth0`就是网卡名，可以使用`ip a`命令查看。  


#### 5. 节省gpu内存的方法  
训练LLM经常会出现内存不够用的情况，一般都是减小句子的长度，这里分享一些其他方法(不是唯一的办法，其他的请自行摸索):  

(1) activation checkingpoint  
这个跟pytorch的`utils.checkpoint`意思一样，在forward之后不保留用于计算梯度的中间结果，而是在backward的时候重新计算一遍，这样会增加计算量，但是可以减小保存中间结果占用的gpu内存空间，属于时间换空间的方法。  
要想这样做就在`configs/ds_config_pp.yml`文件里面设置:  
```yml
use_grad_ckpt: true
```

开启这个选项之后可以支持更长的句子长度，下面同样是设置`micro_batch_size=1`，`global_batch_size=128`，训练20个step看log显示的速度(sample/s)。

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

(2) 使用flash-attention   
flash-attention可以加快qkv的计算速度，而且还能省内存，用过的人都说好，如果你的平台可以运行flash-attention的话，可以在配置文件`configs/ds_config_pp.yml`里面这样设置: 
```yaml
    use_flash_attn: true
```
到2023.8为止，flash-attention还不支持V100，在本项目里面也只支持llama不支持bloom模型。<br>
像baichuan和chatglm这种使用了pytorch的加速attention，就不用设这个选项了，默认就行。<br>

(3) 使用zero的offload  
意思是说，在训练过程中，把一部分gpu内存上的模型参数以及优化器状态等移动到cpu内存上，只有用到的时候再移回gpu内存。这种方法会引入通信延时，就是cpu和gpu之间的通信会导致训练时间变长，属于牺牲了一部分速度换取更多的空间的方法，如果想这样做的话，可以在`configs/ds_config_pp.yml`里面加上下面这个:
```yaml
zero_allow_untested_optimizer: true
zero_force_ds_cpu_optimizer: false
zero_optimization: 
  stage: 1
  offload_optimizer: 
    device: cpu
```

(4) 使用其他优化器  
adamw的一个缺点就是对每个参数都要有param/m/v，也就是要占用三倍参数的存储空间，lion优化器没有这个问题，亲测在我的服务器上使用lion可以在8张v100上训练llama-13b(max_seq_len=128)，如果想试试这个优化器的话，可以在`configs/ds_config_pp.yml`里面把优化器的配置改成这样: 
```yml
optimizer: 
  type: Lion
  params: 
    lr: 2.0e-4
    betas: [0.95, 0.98]
    weight_decay: 2.0e-4
```

注意: 我没有仔细比较过adamw和lion训练好的模型的效果好坏，只是说使用这个可以节省内存，在有限的gpu上训练更大的模型，具体的效果需要使用的人自行把握。另外，这里面使用的训练参数(lr/wd/betas)也是随便设的，可能也需要调一调。    


#### 6. 将训练好的权重转化为huggingface的权重  
运行以下脚本:  
```
    $ python convert_model.py pp_to_hg --input-path /path/to/trained/pp/checkpoint --save-path /path/to/hg
```

到这一步，就可以利用其他项目里面的各种方式加载并且部署了，找到定义模型的地方，像这样手动加载我们自己训练的模型: 
```python
    config = AutoConfig.from_pretrained('/path/to/hg')
    model = AutoModelForCausalLM.from_pretrained('/path/to/hg')
    tokenizer = AutoTokenizer.from_pretrained('/path/to/hg')
```


### 使用训练好的模型权重做推理  

#### 1. 使用deepspeed的推理api
可以参考运行这个代码:  
```
    $ deepspeed --num_gpus 4 --num_nodes 1 demo.py
```
到0.9.2的时候，deepspeed对llama还没有默认支持tensor-parallel，必须手动指定policy才行而且速度也比bloom慢一些，相比之下bloom是默认支持tensor-parallel的。比如使用两张gpu的时候，bloom可以让每张卡占用一半模型的显存，而不指定policy的llama就得两个gpu都占完整模型的显存。  


#### 2. 使用text-generation-inference的推理服务  
注意事项:   
* 需要gpu和驱动的组合可以支持cuda 11.7及以上的版本，我的部署服务器是8张T4的gpu，驱动是515.65.01。  
* 部署llama的话，需要gpu支持flash-attention，到2023.7.1为止，v100是不支持flash-attention的，所以不能用v100部署llama。  
* llama-30b的模型的head数不能被8整除，所以不能使用8张gpu对llama-30b的模型做serving。
* 使用非A100的gpu部署bloom模型，需要加上选项--disable-custom-kernels。

把模型啥的保存到一个目录:  
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

启动服务:  
```
    model_root=./saved_models # 就是上面保存模型使用的./saved_models 
    model_id=save_pretrained_bloom # 上面保存模型时使用的llama_13b_hf/
    num_gpus=8

    $ docker run -d --gpus all --shm-size 64g -p 8082:80 -v $model_root:/data ghcr.io/huggingface/text-generation-inference:0.8 --num-shard $num_gpus --model-id $model_id # --disable-custom-kernels
```

调用服务: 
```
    url=127.0.0.1:8082/generate # 运行完统一返回整个结果
    # url=127.0.0.1:8082/generate_stream # 流式返回结果，生成一个返回一个

    $ curl ${url} \
        -X POST \
        -d '{"inputs":"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n简化以下段落，使其更易理解\n\n### Input:\n尽管人们普遍认为互联网使我们能够与世界各地的人联系，但仍有一些人不熟悉其基本功能，不理解为什么它变得如此普遍，或者它的真正能力是什么。\n\n### Response:","parameters":{"max_new_tokens":17}}' \
        -H 'Content-Type: application/json'
```

这个性能还蛮好的，亲测可在1张T4上部署7b大小的模型，而且速度很快。  


### 预训练权重  

本来想训个东西放出来给大家玩的，无奈现在没有算力，等我有算力的时候再说吧。  



### 最后  
如果你发现代码里面有任何错误，或者有更好的实现方式，请开issue告诉我，方便及时修正，另外如果又出了什么新的工具或者新玩法或者高质量数据集啥的，也欢迎提issue分享，感激不尽。 <br><br><br><br>



================== 分割线 ==========================

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
### 别看了，后面没有了
<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
### 真的没有了
<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
### 好吧。。。。既然有人翻到这里。。。

这个项目出来这么久了也没几个星星，估计是没什么人关注，那就扯点没用的吧，毕竟生活压力大需要发泄。反正也没人看。Positive negative also no person look.<br><br><br><br>
                                                            ![image](https://github.com/CoinCheung/gdGPT/assets/22693362/003586cd-6069-4dfd-9b70-a62fed31052d)
                                                            <br><br><br>



### 下面的内容都是胡说八道  


貌似大家的开源分享意愿还不是很强，各大公司都宣布自己遥遥领先，股价也是纷纷上涨，但是事后好像也没有放出数据啥的或者只放出来了一部分，感觉大部分人都在不动声色的收集别人开源的东西，但是又没怎么分享自己的东西出来给别人用。。。知道有那种特别擅长搜集信息的人，把各家分享的优质资源整合起来，然后包装一下就宣布自己的单位做了个很牛逼的东西，我觉得这种做法一点问题都没有，不需要上纲上线批判啥的，毕竟大家都要吃饭的，追名逐利也是人之常情，就是希望能分享一下中间过程的心得体会还有开源一下整合的数据啥的就更好了。不是很理解为啥好多人忿忿不平的指责openai不开源chatgpt，这是被照顾得太好了，被惯得认为别人的都欠他们的?<br>

你说要想实现所谓的人工智能，真的就只有依赖海量算力把模型做大这一条路可以走吗，让我用阴谋论往坏处想一想，因为总要有人来当这个又蠢又坏的讨人嫌，总要有人出来说点让人不爱听的话。。。冷战的时候，如果美国并没有真的实现登月，而是放出一个假消息引导苏联在一件不是很紧要的事情上消耗大量的国力，这样就可以慢慢的拖垮苏联。。。假如我手下有全世界最顶尖的那一批天才科学家为我工作，而且我手里有其他国家都没办法生产的芯片，那么我就会让这些科学家把最前沿的科研成果向依赖芯片算力的方向推动，这样我就可以使劲卖芯片来赚钱了，即使在不需要把模型做得超级大的情况下也有办法实现我们想象中的那种人工智能，我也还是会这么干，反正其他所有人都做不出来更好的。。<br>

我觉得判断一个工作的价值，要跟这个工作之前相比，而不是跟后面的工作比较。比如resnet提出好多年之后，又有人出来说residual没什么了不起的，看明白别人的东西之后产生了你上你也行的错觉，实际上有好多人就只会用开源代码跑自己的数据，连超参数都设不明白的人也有不少吧。我只能说可惜这些人生不逢时，要是让他们早生几年，就能赶在何凯明之前提出resnet了。要是看了别人毫无保留的分享之前你就想到可以怎么做算是英雄所见略同，学会了别人的东西之后再跳出来摆谱，感觉像个大尾巴狼。好多人的那种对自己实力的自信和优越感，更多是来自于知识面，看到了大部分人没看到的开源代码里面的trick，或者看了一些冷门但是效果好的工作，所以领先了一些人，我想这只能证明搜集情报的能力，而不是创造力，更像是一种钻营。<br>

我认为对于AI的应用要谨慎。虽然人们常说技术无罪，应该惩罚杀人犯而不是禁止使用刀子，但是也有可能会像小孩子玩火那样引起火灾吧。假如你做了一个AI客服可以给老板省100块钱，然后老板奖励你10块钱，你就会觉得这是你的本事，那些失去工作的人工客服都是活该。你认为你在行善，但是别人认为你在作恶。因为你只是改变的存量的分配方式，加速了强者对弱者的剥削，而没有增加社会总价值，就像短视频或者直播带货那种不产生实际价值。人类和AI的关系更像是成年人和刀子的关系，还是更像小孩子和火的关系呢。我是觉得应该利用高科技去做人类做不到的事，而不是总想着代替人工。每年高空作业或者水下作业的工人意外死亡的不在少数，想办法避免让工人去危险的环境下工作，要比做个AI客服更有意义吧。毕竟一个直观的想法是，一切的努力都是为了让世界变得更好，而不是让人类相互捅刀子最后走向凋零。<br>


以上内容全是胡说八道，而且没有任何依据，认真你就输了。
