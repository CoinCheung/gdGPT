
## 使用deepspeed对bloomz进行finetune

这个项目没有什么理论上的创新，没有提出茴香豆的新写法，也没发明什么新工具，仅仅是基于现有的方法和库提供一套简洁易扩展的代码，可以在8张v100服务器上训练7b的模型(对全部模型参数做full-finetune的那种训练)，速度比zero3方法更快，并且支持更长的输入序列长度。    
在我的8张v100的服务器上，当句子长度为1024时，bloom-7b的训练速度为7 sample/s，llama-7b的训练速度为8 sample/s。如果使用gradient checkpoint可以支持更长的输入句子长度。


### 我的环境  
* Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
* 512G cpu memory
* v100 x 8
* ubuntu 18.04 
* python 3.8.12
* driver 515.65.01 
* cuda11.6 + cudnn8 
* deepspeed==0.9.2 
* torch==1.13.1
* sentencepiece
* protobuf==3.20.0 (python pip install)


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
    # INPUT=/path/to/models # 使用save_pretrained保存的模型和tokenizer，一定要包括tokenizer
    SAVE_PATH=./saved_bloomz_7b1_mt_pp

    python convert_model.py hg_to_pp --input-path $INPUT --save-path $SAVE_PATH
```



#### 3. 训练  
(1) 单机训练  
可以运行这个命令:  
```
    $ deepspeed train_ds.py --config configs/ds_config_pp.json
```

(2) 多机训练  
当8张v100不太够用的时候，就得用多机联机训练。首先需要安装pdsh，然后配置一下ssh服务让不同结点之间可以使用ssh免密登陆，再根据ssh结点名配置编辑hostfile，用下面的命令来启动，这个过程需要保证每台服务器上的代码和各种文件**完全相同**:  
```
    $ deepspeed --hostfile ./hostfile train_ds.py --config ds_configs/ds_config_pp.json
```
hostfile的格式可以参考这个示例的[hostfile](./hostfile)文件。  

经过实验和推算，当打开`gradient checkpointing`并且将`max_seq_len`设为2048时，使用AdamW优化器，训练llama-13b模型需要14张v100，训练llama-30b需要31张v100，训练llama-65b需要80张v100。  

请注意:  
* 如果你在docker环境做多机训练的话，需要在启动docker时加上`--network=host`选项。  
* 如果在多机并行的时候遇到NCCL的问题，需要加上一个环境变量用来指定网卡名:  
```
    echo "NCCL_SOCKET_IFNAME=eth0" > ./.deepspeed_env
```
这里面的`eth0`就是网卡名，可以使用`ip a`命令查看。  


#### 4. 节省gpu内存的方法  
训练LLM经常会出现内存不够用的情况，一般都是减小句子的长度，这里分享一些其他方法(不是唯一的办法，其他的请自行摸索):  

(1) activation checkingpoint  
这个跟pytorch的`utils.checkpoint`意思一样，在forward之后不保留用于计算梯度的中间结果，而是在backward的时候重新计算一遍，这样会增加计算量，但是可以减小保存中间结果占用的gpu内存空间，属于时间换空间的方法。  
要想这样做就在`configs/ds_config_pp.json`文件里面设置:  
```json
"use_grad_ckpt": true
```

(2) 使用zero的offload  
意思是说，在训练过程中，把一部分gpu内存上的模型参数以及优化器状态等移动到cpu内存上，只有用到的时候再移回gpu内存。这种方法会引入通信延时，就是cpu和gpu之间的通信会导致训练时间变长，属于牺牲了一部分速度换取更多的空间的方法，如果想这样做的话，可以在`configs/ds_config_pp.json`里面加上下面这个:
```json
"zero_force_ds_cpu_optimizer": false,
"zero_optimization": {
    "stage": 1,
    "offload_param": {
        "device": "cpu",
        "pin_memory": true
    },
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
    },
},
```

(3) 使用其他优化器  
adamw的一个缺点就是对每个参数都要有param/m/v，也就是要占用三倍参数的存储空间，lion优化器没有这个问题，亲测在我的服务器上使用lion可以在8张v100上训练llama-13b(max_seq_len=128)，如果想试试这个优化器的话，可以在`configs/ds_config_pp.json`里面把优化器的配置改成这样: 
```json
"optimizer": {
    "type": "Lion",
    "params": {
      "lr": 2e-4,
      "betas": [
        0.9,
        0.999
      ],
      "use_triton": true,
      "weight_decay": 2e-4
    }
},
```

注意: 我没有仔细比较过adamw和lion训练好的模型的效果好坏，只是说使用这个可以节省内存，在有限的gpu上训练更大的模型，具体的效果需要使用的人自行把握。另外，这里面使用的训练参数(lr/wd/betas)也是随便设的，可能也需要调一调。    


#### 5. 模型pipeline的设置方法  
在`configs/ds_config_pp.json`里面有这样的配置选项:  
```json
"model_topo": {
    "process_topology": {
        "axis": ["pipe", "data"],
        "dims": [8, 1]
    },
    "parts": [1, 5, 5, 5, 5, 5, 5, 1] 
},
```
这个表示一共有`8x1=8`张gpu，并且8张gpu上只有一个模型，如果是`dims: [8,2]`的话，就表示一共有`8x2=16`张gpu，并且每8张gpu上有一个模型，16张gpu上共有两个模型。  
另外就是`parts`表示一个模型在8张gpu上是怎么分配的，`bloom-7b`的模型共有30个transformer的block，加上两端的embedding共有32个block，`parts: [1, 5, 5, 5, 5, 5, 5, 1]`表示第一张和最后一张gpu上各有1个block(按顺序应该是embedding)，中间的6张gpu上每张有5个block(transformer的block)。  

友情提示: block的分布方式除了要考虑gpu内存之外，还得考虑每张卡的计算负载，因为训练速度决定于最慢的那张gpu，所以要尽量避免某一个gpu计算量比其他gpu大很多的情况。  


#### 6. 将训练好的权重转化为huggingface的权重  
运行以下脚本:  
```
    $ python convert_model.py pp_to_hg --input-path /path/to/pp/checkpoint --save-path /path/to/hg
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
    model_root=/data/models # 把模型使用save_pretrained的方式，保存到这个目录的一个子目录里面
    model_id=save_pretrained_bloom # 这个就是上面的子目录的名字
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

本来想训个东西放出来给大家用的，无奈现在没有算力，等我有算力的时候再说吧。  



### 最后  
如果你发现代码里面有任何错误，或者有更好的实现方式，请开issue告诉我，方便及时修正，另外如果又出了什么新的工具或者新玩法或者高质量数据集啥的，也欢迎提issue分享，感激不尽。  



================== 分割线 ==========================

### 下面的内容都是胡说八道  
到了夜深人静的时候，就时常有些奇怪的想法进入脑海，有的光明有的阴暗还有各种异想天开的意淫啥的。   

貌似大家的开源分享意愿还不是很强，各大公司都宣布自己遥遥领先，股价也是纷纷上涨，但是事后好像也没有放出数据啥的或者只放出来了一部分，感觉大部分人都在不动声色的收集别人开源的东西，但是又没怎么分享自己的东西出来给别人用。。。知道有那种特别擅长搜集信息的人，把各家分享的优质资源整合起来，然后包装一下就宣布自己的单位做了个很牛逼的东西，我觉得这种做法一点问题都没有，不需要上纲上线批判啥的，毕竟大家都要吃饭的，追名逐利也是人之常情，就是希望能分享一下中间过程的心得体会还有开源一下整合的数据啥的就更好了。。。  

你说要想实现所谓的人工智能，真的就只有依赖海量算力把模型做大这一条路可以走吗，让我用阴谋论往坏处想一想，因为总要有人来当这个又蠢又坏的讨人嫌，总要有人出来说点让人不爱听的话。。。冷战的时候，如果美国并没有真的实现登月，而是放出一个假消息引导苏联在一件不是很紧要的事情上消耗大量的国力，这样就可以慢慢的拖垮苏联。。。假如我手下有全世界最顶尖的那一批天才科学家为我工作，而且我手里有其他国家都没办法生产的芯片，那么我就会让这些科学家把最前沿的科研成果向依赖芯片算力的方向推动，这样我就可以使劲卖芯片来赚钱了，即使在不需要把模型做得超级大的情况下也有办法实现我们想象中的那种人工智能，我也还是会这么干，反正其他所有人都做不出来更好的。。。 

以上内容全是胡说八道，而且没有任何依据，认真你就输了。
