
import json
import os
import re
import random


def get_texts_v1(max_seq_len=512):
    with open('datasets/zhaoliang_train.json', 'r') as fr:
        lines = fr.read().splitlines()
        texts = []
        for l in lines:
            try:
                l = json.loads(l)
            except:
                print(l)
                continue
            l = '<s>' + l['input'] + '</s>' + l['target'] + '</s>'
            if len(l) > max_seq_len: continue
            texts.append(l)
    random.seed(123)
    random.shuffle(texts)
    texts = texts[:1000]
    return texts



sep = '@@##sep##@@'
def get_texts_novel_instruct_220(max_seq_len=512, tokenizer=None):
    with open('datasets/novel_write_230.jsonl', 'r') as fr:
        lines = fr.read().splitlines()
    texts = []
    for l in lines:
        try:
            l = json.loads(l)
        except:
            print(l)
            continue
        prompts = re.sub('\n##\n\n$', '', l['prompt'])
        completion = re.sub('##', '', l['completion'])

        l = '<s>' + prompts + '</s>' + completion + '</s>'
        if not tokenizer is None:
            ipids = tokenizer([l,])["input_ids"][0]
            if len(ipids) > max_seq_len: continue
        else:
            if len(l) > max_seq_len: continue
        texts.append(l)
    with open('datasets/novel_write_230.txt', 'w') as fw:
        fw.write(f'{sep}'.join(texts))
    return texts


def get_texts_novel_instruct_json(max_seq_len=512, tokenizer=None):

    with open('datasets/novel_512_40w_instruct_gen_dev.json', 'r') as fr:
        jobj = json.load(fr)
    res = []
    for ob in jobj:
        header = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'
        instruct = '### Instruction:\n' + ob['instruct'] + '\n\n'
        inp = '### Input:\n' + ob['input'] + '\n\n' if 'input' in ob else ''
        resp = '### Response:\n'
        prompt = header + instruct + inp + resp
        outp = ob['output'] + '\n'
        res.append((prompt, outp))

    return res



def get_texts_multi_conversation_json(max_seq_len=512, tokenizer=None):
    with open('datasets/processed_multiturn_chat_0.8M.json.dev', 'r') as fr:
        jobj = json.load(fr)
    role_map = {'ask': 'Human', 'ans': 'Assistent'}
    res = []
    for ob in jobj:
        one_res = []
        header = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. \n\n"
        res_txt = header
        for role, txt in ob['rounds']:
            if role == 'ask':
                res_txt = res_txt + f'### {role_map[role]}: {txt} \n'

            elif role == 'ans':
                prefix = f'### {role_map[role]}: '
                res_txt = res_txt + prefix
                one_res.append(res_txt)

                res_txt = res_txt + f'{txt} ' # 这个空格一定要有
                res_txt = res_txt + tokenizer.eos_token + '\n'
        one_res.append(res_txt)

        #  for el in one_res:
        #      print(el)
        #      print('=============')
        #      print('=============')
        #
        #  _ = input()

        res.append(one_res)
    return res


def get_texts_ref_gpt_json(max_seq_len=512, tokenizer=None):
    with open('datasets/processed_RefGPT-Dataset-V1-CN.json.dev', 'r') as fr:
        jobj = json.load(fr)
    role_map = {'ask': 'Instruction', 'ans': 'Response'}
    res = []
    for ob in jobj:
        one_res = []
        head_ref_txt = f"Generate responses to the given instruction series according to the reference text. \n\n### Reference: \n{ob['reference']}\n\n"
        res_txt = head_ref_txt
        for role, txt in ob['rounds']:
            if role == 'ask':
                res_txt = res_txt + f'### {role_map[role]}: {txt} \n'

            elif role == 'ans':
                prefix = f'### {role_map[role]}: '
                res_txt = res_txt + prefix
                one_res.append(res_txt)

                res_txt = res_txt + f'{txt} ' # 这个空格一定要有
                res_txt = res_txt + tokenizer.eos_token + '\n'
        one_res.append(res_txt)

        #  for el in one_res:
        #      print(el)
        #      print('=============')
        #      print('=============')
        #
        #  _ = input()

        res.append(one_res)
    return res


def get_texts_novel_lm_1w():
    pth = f'./datasets/novel_512_1w.txt'
    pth = f'./datasets/novel_data.txt'
    with open(pth, 'r') as fr:
        texts = fr.read().split('@@##sep##@@')

    res = ['<s>' + el + '</s>' for el in texts]
    #  res = []
    #  for el in texts:
    #    l = len(el)
    #    pos = int(random.uniform(50, l - 50))
    #    el = '<s>' + el[:pos] + '</s>' + el[pos:] + '</s>'
    #    res.append(el)
    return res


if __name__ == '__main__':
    from transformers import AutoTokenizer
    model_name = 'bigscience/bloom-7b1'
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    #  get_texts_novel_instruct(max_seq_len=768, tokenizer=tokenizer)
    get_texts_multi_conversation_json(max_seq_len=768, tokenizer=tokenizer)

