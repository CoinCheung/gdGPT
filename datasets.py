
import os
import os.path as osp
import re
import random
import json
from bisect import bisect
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, LlamaTokenizer



## TODO: 句末要加上</s>，让模型学会停止，开头加不加都行，就不用加了


def truncate_and_pad_left(input_ids, att_mask, labels, max_seq_len, tokenizer):
    # truncate to max_seq_len
    input_ids = input_ids[:max_seq_len]
    att_mask = att_mask[:max_seq_len]
    labels = labels[:max_seq_len]

    # pad to left
    len_pad = max_seq_len - input_ids.size(0)
    if len_pad > 0:
        pad_token_id = tokenizer.pad_token_id
        pad_inp = torch.zeros(len_pad, dtype=torch.long).fill_(pad_token_id)
        pad_att = torch.zeros(len_pad, dtype=torch.long)
        pad_lb = torch.zeros(len_pad, dtype=torch.long).fill_(-100) # ignore pad label
        input_ids = torch.cat([pad_inp, input_ids], dim=0)
        att_mask = torch.cat([pad_att, att_mask], dim=0)
        labels = torch.cat([pad_lb, labels], dim=0)

    inputs = torch.cat([input_ids.unsqueeze(-1), att_mask.unsqueeze(-1)], dim=-1)

    return inputs.clone(), labels.clone()


def process_given_sentence(tokenizer, txt, ignore_label):
    toks = tokenizer(txt, add_special_tokens=False,
                       return_tensors="pt",
                       padding=False, truncation=False)
    hr_inp, hr_attm = toks.input_ids[0], toks.attention_mask[0]
    hr_lb = hr_inp.clone()
    if ignore_label: hr_lb.fill_(-100)
    return hr_inp, hr_attm, hr_lb


## TODO: 1. 根据长度删除后面超长的rounds
def process_conversation_rounds(tokenizer, rounds, role_map, ignore_known):
    rounds_inp, rounds_attm, rounds_lb = [], [], []
    for role, txt in rounds:
        if role == 'ask':
            txt = f'### {role_map[role]}: {txt} \n'
            r_inp, r_attm, r_lb = process_given_sentence(tokenizer, txt, ignore_known)
            rounds_inp.append(r_inp)
            rounds_attm.append(r_attm)
            rounds_lb.append(r_lb)
        elif role == 'ans':
            prefix = f'### {role_map[role]}: '
            txt = f'{txt} ' # 这个空格不能少
            txt = txt + tokenizer.eos_token + '\n'
            p_inp, p_attm, p_lb = process_given_sentence(tokenizer, prefix, ignore_known)
            t_inp, t_attm, t_lb = process_given_sentence(tokenizer, txt, False)
            rounds_inp += [p_inp, t_inp]
            rounds_attm += [p_attm, t_attm]
            rounds_lb += [p_lb, t_lb]
        else:
            raise NotImplementedError
    return rounds_inp, rounds_attm, rounds_lb


def parse_pretrain_sample(tokenizer, ob, max_seq_len):
    '''
        ob = {
            "type": "pretrain",
            "text": "我只想说懂得都懂，不懂的我也不多解释，毕竟自己知道就好，细细品吧。你们也别来问我怎么了，利益牵扯太大，说了对你我都没好处，当不知道就行了，其余的我只能说这里面水很深，牵扯到很多东西。详细情况你们自己是很难找的，网上大部分已经删除干净了，所以我只能说懂得都懂。懂的人已经基本都获利上岸什么的了，不懂的人永远不懂，关键懂的人都是自己悟的，你也不知道谁是懂的人也没法请教，大家都藏着掖着生怕别人知道自己懂事，懂了就能收割不懂的，你甚至都不知道自己不懂。只是在有些时候，某些人对某些事情不懂装懂，还以为别人不懂。"
        }
    '''
    txt = ob['text'] # 这个还是不加bos/eos了，反正pretrain，其实如果严格小于max_seq_len的话，可以加一个</s>，但是因为句子是截断的，所以也不一定最后一个字后面就一定是eos，所以干脆就不加了，这样在pretrain的时候，就学不会什么时候eos

    ## 这里必须是padding='max_length'，不然会出现梯度形状与tensor形状不一致的问题
    ## max_length means limit to max_length, or True/longest which means padding to longest of the batch, or False/no_not_pad which means no padding
    inputs = tokenizer(txt, add_special_tokens=False,
                       return_tensors="pt",
                       padding='max_length',
                       truncation=True, max_length=max_seq_len)
    input_ids = inputs["input_ids"][0]
    att_mask = inputs.attention_mask[0]
    labels = input_ids.clone()
    inputs = torch.cat([input_ids.unsqueeze(-1), att_mask.unsqueeze(-1)], dim=-1)
    labels[~att_mask.bool()] = -100 # 这样就不会考虑pad部分的loss了
    return inputs, labels


def parse_instruct_sample(tokenizer, ob, max_seq_len, ignore_known=False):
    '''
    数据格式
        ob = {
            "type": "instruct",
            "instruct": "补充下面横线上的内容",
            "input": "再多看一眼就会爆炸，________",
            "output": "再。。。再靠近点快被融化?"
        }
        或者:
        ob = {
            "type": "instruct",
            "instruct": "写一篇拍老板马屁的文章，题目是《xx的十宗罪》。要求以批评的语气来写，看起来像是在批评其实说的都是作为资本家还不够黑心之类的，比如老板的缺点就是工作太辛苦啥的，让老板看完眼前一亮然后发到公司内部的员工论坛上，之后各大媒体争相报道，外面的人都跟着高潮了。",
            "output": "你要是没事干去村头把粪挑了"
        }

    可以设置是否把prompt的label设为ignore_index，把pad的label设为ignore_index
    '''
    # create prompt and output in text
    header = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruct = '### Instruction:\n' + ob['instruct'] + '\n\n'
    inp = ''
    if 'input' in ob and ob['input'].strip() != '':
        inp = '### Input:\n' + ob['input'] + '\n\n'
    resp = '### Response:\n'
    prompt = header + instruct + inp + resp
    outp = ob['output'] + '\n' + tokenizer.eos_token

    # tokenizer prompt and output separately,
    p_inp, p_attm, p_lb = process_given_sentence(tokenizer, prompt, ignore_known)
    o_inp, o_attm, o_lb = process_given_sentence(tokenizer, outp, False)

    input_ids = torch.cat([p_inp, o_inp], dim=0)
    att_mask = torch.cat([p_attm, o_attm], dim=0)
    labels = torch.cat([p_lb, o_lb], dim=0)

    res = truncate_and_pad_left(input_ids, att_mask, labels, max_seq_len, tokenizer)
    inputs, labels = res
    labels[-1] = tokenizer.eos_token_id

    return inputs, labels



def parse_conversation_sample(tokenizer, ob, max_seq_len, ignore_known=False):
    '''
    数据格式
        ob = {
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
        }
    可以设置把prompt和human的部分的label设为ignore_index，把pad的label设为ignore_index
    '''
    rounds_inp, rounds_attm, rounds_lb = [], [], []
    # process header part
    header = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. \n\n"
    h_inp, h_attm, h_lb = process_given_sentence(tokenizer, header, ignore_known)

    # process rounds
    role_map = {'ask': 'Human', 'ans': 'Assistent'}
    res_rounds = process_conversation_rounds(tokenizer, ob['rounds'],
                                             role_map, ignore_known)

    # combine
    input_ids = torch.cat([h_inp, ] + res_rounds[0], dim=0)
    att_mask = torch.cat([h_attm, ] + res_rounds[1], dim=0)
    labels = torch.cat([h_lb, ] + res_rounds[2], dim=0)

    res = truncate_and_pad_left(input_ids, att_mask, labels, max_seq_len, tokenizer)
    inputs, labels = res
    labels[-1] = tokenizer.eos_token_id

    return inputs, labels


def parse_ref_qa_sample(tokenizer, ob, max_seq_len, ignore_known=False):
    '''
    数据格式
        ob = {
            "type": "ref_qa",
            "reference": "一掐脖子就翻白眼一松手就吹牛逼，早有布局遥遥领先拳打谷歌脚踢微软千秋万代一统江湖",
            "rounds": [
                ["ask", "这段话有几个字"],
                ["ans", "100个字"],
                ["ask", "多少汉字多少英文"],
                ["ans", "你不会自己看?"],
            ]
        }
    可以设置把prompt和human的部分的label设为ignore_index，把pad的label设为ignore_index
    '''

    # create header part and reference text
    head_ref_txt = f"Generate responses to the given instruction series according to the reference text. \n\n### Reference: \n{ob['reference']}\n\n"
    hr_inp, hr_attm, hr_lb = process_given_sentence(tokenizer, head_ref_txt, ignore_known)

    # process instruct rounds
    role_map = {'ask': 'Instruction', 'ans': 'Response'}
    res_rounds = process_conversation_rounds(tokenizer, ob['rounds'],
                                             role_map, ignore_known)

    # combine
    input_ids = torch.cat([hr_inp, ] + res_rounds[0], dim=0)
    att_mask = torch.cat([hr_attm, ] + res_rounds[1], dim=0)
    labels = torch.cat([hr_lb, ] + res_rounds[2], dim=0)

    res = truncate_and_pad_left(input_ids, att_mask, labels, max_seq_len, tokenizer)
    inputs, labels = res
    labels[-1] = tokenizer.eos_token_id

    return inputs, labels


def get_sample_from_jobj(tokenizer, ob, max_seq_len):
    '''
    '''
    stype = ob['type']
    if stype == 'pretrain':
        res = parse_pretrain_sample(tokenizer, ob, max_seq_len)
    elif stype == 'instruct':
        res = parse_instruct_sample(tokenizer, ob, max_seq_len, ignore_known=True)
    elif stype == 'conversation':
        res = parse_conversation_sample(tokenizer, ob, max_seq_len, ignore_known=True)
    elif stype == 'ref_qa':
        res = parse_ref_qa_sample(tokenizer, ob, max_seq_len, ignore_known=True)
    else:
        raise NotImplementedError

    return res



## NOTE:
## llama是byte编码的，bloom是字符编码的，同样是100个汉字的情况下，llama输出会大于100，bloom会输出少于100，llama不会出现unk的情况.
## 一般来说tokenizer的时候，看情况在后面加上eos就行了，不用在前面加上bos，反正lm也不需要
def get_tokenizer(model_name):
    #  # add <pad> to left of the input seq, for batch inference
    #  if re.search('llama', model_name):
    #      tokenizer = LlamaTokenizer.from_pretrained(model_name,
    #              add_bos_token=False, # 是否自动在前面加一个bos
    #              add_eos_token=False, # 是否自动在后面加一个eos
    #              bos_token = '<s>',
    #              eos_token = '</s>',
    #              padding_side='left')
    #      tokenizer.pad_token = '<unk>' # 让pad为啥都可以，因为qkv的时候都不考虑pad的部分，计算loss的时候也不考虑pad部分，整体相当于没有用到pad部分，仅仅是在计算的时候凑数用，至于被当成pad的token，在qkv的时候还是用到了，并且计算loss时候也用到了，不影响
    #  elif re.search('bloom', model_name):
    #      tokenizer = AutoTokenizer.from_pretrained(model_name,
    #              add_bos_token=False,
    #              add_eos_token=False,
    #              padding_side='left')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer




class TextDataSet(Dataset):

    def __init__(self, data_file, model_name, max_seq_len=1024):

        with open(data_file, 'r') as fr:
            self.samples = json.load(fr)

        self.tokenizer = get_tokenizer(model_name)
        self.max_seq_len = max_seq_len

    def __getitem__(self, ind):
        return get_sample_from_jobj(self.tokenizer, self.samples[ind], self.max_seq_len)

    def __len__(self):
        return len(self.samples)

    def save_tokenizer(self, save_path):
        self.tokenizer.save_pretrained(save_path)
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        tokenizer = AutoTokenizer.from_pretrained(save_path)



class TextDataSetShards(Dataset):

    def __init__(self, froot, model_name, max_seq_len=1024):
        '''
        froot是一个目录，里面都是text文件，这个类主要是实现了一个类似于cacheline的功能，把一个大文件分成许多小的text文件，这样可以每次读一个小的text文件，而不是把整个文本都读进来

        解析的方法就是读进来，然后使用一个separator拆成长度不等的句子就行了。

        这个需要有一个get_shard_lens的属性，返回一个list，里面是每一个小text文件里面的样本个数，有了这个再配合自己实现的sampler，就可以每次shuffle之后让同一个text文件里面的样本的index都连着，这样就不用每取一个样本都重新读一遍cache line了
        '''

        fnames = [el for el in os.listdir(froot) if re.search('\.json$', el)]
        self.fpaths = [os.path.join(froot, fn) for fn in fnames]
        self.length = 0

        self.shard_lens, self.shard_milestones = [], [0,]
        for f in self.fpaths:
            txts = get_texts_from_file(f)
            self.length += len(txts)
            self.shard_lens.append(len(txts))
            self.shard_milestones.append(self.length)
        self.f_pre = None

        self.tokenizer = get_tokenizer(model_name)
        self.max_seq_len = max_seq_len

    @property
    def get_shard_lens(self):
        return self.shard_lens

    def __getitem__(self, ind):
        f_ind = bisect(self.shard_milestones, ind)
        pth = self.fpaths[f_ind]
        if not pth == self.f_pre:
            ## 可以考虑在这个地方把self.f_pre删除，这样就可以resume状态了
            #  if self.f_pre and osp.exists(self.f_pre): os.remove(self.f_pre)
            self.cacheline = get_texts_from_file(pth)
            self.f_pre = pth
        ind -= self.shard_milestones[f_ind]

        return get_inp_labels(self.tokenizer, self.texts[ind], self.max_seq_len)

    def __len__(self):
        return self.length

    def save_tokenizer(self, save_path):
        self.tokenizer.save_pretrained(save_path)
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        tokenizer = AutoTokenizer.from_pretrained(save_path)

