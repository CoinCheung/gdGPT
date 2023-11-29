
import os
import os.path as osp
import sys
import argparse
import re
import torch
from pprint import pprint

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig



parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")
download_parser = subparsers.add_parser('download')
download_parser.add_argument('--model-name', default='bigscience/bloomz-560m')
download_parser.add_argument('--save-path', default='./saved_model/')

hg2ppp_parser = subparsers.add_parser('hg_to_pp')
hg2ppp_parser.add_argument('--input-path', default=None)
hg2ppp_parser.add_argument('--save-path', default='./pp_model/')

pp2hg_parser = subparsers.add_parser('pp_to_hg')
pp2hg_parser.add_argument('--input-path', default='./pp_model/')
pp2hg_parser.add_argument('--save-path', default='./saved_model/')
args = parser.parse_args()


def save_py_file(obj, save_path):
    src = sys.modules[obj.__module__].__file__
    dst = osp.join(save_path, osp.basename(src))
    with open(src, 'r') as fr:
        txt = fr.read()
    txt = re.sub('(?<=from\s)\.\.\.(?=\s)', 'transformers', txt)
    txt = re.sub('(?<=from\s)\.\.\.(?=[A-Za-z])', 'transformers.', txt)
    with open(dst, 'w') as fw:
        fw.write(txt)


### bloom

def convert_bloom_hg2pp(state):
    res = {
        0: {
            'word_embeddings.weight': state['transformer.word_embeddings.weight'],
            'word_embeddings_layernorm.weight': state['transformer.word_embeddings_layernorm.weight'],
            'word_embeddings_layernorm.bias': state['transformer.word_embeddings_layernorm.bias'],
        },
    }

    ind_last = -1
    for k,v in state.items():
        if not re.search('^transformer.h.', k): continue
        k = re.sub('^transformer.h.', '', k)
        ind = int(re.search('^\d+', k).group())
        k = re.sub('^\d+\.', '', k)
        ind += 1
        if not ind in res: res[ind] = {}
        res[ind][k] = v
        ind_last = max(ind_last, ind)

    ind_last += 1
    last = {
        ind_last: {
            'word_embeddings.weight': state['transformer.word_embeddings.weight'],
            'word_embeddings_layernorm.weight': state['transformer.ln_f.weight'],
            'word_embeddings_layernorm.bias': state['transformer.ln_f.bias'],
        },
    }
    if 'lm_head.weight' in state.keys():
        last[ind_last]['word_embeddings.weight'] = state['lm_head.weight']
    res.update(last)
    return res


def convert_bloom_pp2hg(pts):
    states = {}
    for ind, pt in pts[1:-1]:
        tmp_state = torch.load(pt, map_location='cpu')
        for k,v in tmp_state.items():
            k = f'transformer.h.{ind - 1}.{k}'
            states[k] = v

    first_states = torch.load(pts[0][1], map_location='cpu')
    last_states = torch.load(pts[-1][1], map_location='cpu')
    states['transformer.word_embeddings.weight'] = first_states['word_embeddings.weight']
    states['transformer.word_embeddings_layernorm.weight'] = first_states['word_embeddings_layernorm.weight']
    states['transformer.word_embeddings_layernorm.bias'] = first_states['word_embeddings_layernorm.bias']
    states['lm_head.weight'] = last_states['word_embeddings.weight']
    states['transformer.ln_f.weight'] = last_states['word_embeddings_layernorm.weight']
    states['transformer.ln_f.bias'] = last_states['word_embeddings_layernorm.bias']
    return states


### llama

def convert_llama_hg2pp(state):
    res = {
        0: {
            'embed_tokens.weight': state['model.embed_tokens.weight'],
        },
    }
    ind_last = -1
    for k,v in state.items():
        if not re.search('^model.layers.', k): continue
        k = re.sub('^model.layers.', '', k)
        ind = int(re.search('^\d+', k).group())
        k = re.sub('^\d+\.', '', k)
        ind += 1
        if not ind in res: res[ind] = {}
        res[ind][k] = v
        ind_last = max(ind_last, ind)

    ind_last += 1
    last = {
        ind_last: {
            'norm.weight': state['model.norm.weight'],
            'embed_tokens.weight': state['lm_head.weight'],
        },
    }
    res.update(last)
    return res


def convert_llama_pp2hg(pts):
    states = {}
    for ind, pt in pts[1:-1]:
        tmp_state = torch.load(pt, map_location='cpu')
        for k,v in tmp_state.items():
            k = f'model.layers.{ind - 1}.{k}'
            states[k] = v

    first_states = torch.load(pts[0][1], map_location='cpu')
    last_states = torch.load(pts[-1][1], map_location='cpu')
    states['model.embed_tokens.weight'] = first_states['embed_tokens.weight']
    #  states['lm_head.weight'] = last_states['word_embeddings.weight']
    states['model.norm.weight'] = last_states['norm.weight']
    states['lm_head.weight'] = last_states['embed_tokens.weight']
    return states


### baichuan-2

def convert_baichuan2_7b_hg2pp(state):
    res = {
        0: {
            'embed_tokens.weight': state['model.embed_tokens.weight'],
        },
    }
    ind_last = -1
    for k,v in state.items():
        if not re.search('^model.layers.', k): continue
        k = re.sub('^model.layers.', '', k)
        ind = int(re.search('^\d+', k).group())
        k = re.sub('^\d+\.', '', k)
        ind += 1
        if not ind in res: res[ind] = {}
        res[ind][k] = v
        ind_last = max(ind_last, ind)

    ind_last += 1
    last = {
        ind_last: {
            'norm.weight': state['model.norm.weight'],
            'lm_head.weight': state['lm_head.weight'],
        },
    }
    res.update(last)
    return res



def convert_baichuan2_7b_pp2hg(pts):
    states = {}
    for ind, pt in pts[1:-1]:
        tmp_state = torch.load(pt, map_location='cpu')
        for k,v in tmp_state.items():
            k = f'model.layers.{ind - 1}.{k}'
            states[k] = v

    first_states = torch.load(pts[0][1], map_location='cpu')
    last_states = torch.load(pts[-1][1], map_location='cpu')
    states['model.embed_tokens.weight'] = first_states['embed_tokens.weight']
    #  states['lm_head.weight'] = last_states['word_embeddings.weight']
    states['model.norm.weight'] = last_states['norm.weight']
    states['lm_head.weight'] = last_states['lm_head.weight']
    return states


### chatglm3-6b

def convert_chatglm3_6b_hg2pp(state):
    res = {
        0: {
            'word_embeddings.weight': state['transformer.embedding.word_embeddings.weight'],
        },
    }

    rotery_states = {re.sub('^transformer.', '', k):v
            for k,v in state.items()
            if re.search('^transformer.rotary_pos_emb', k)}

    ind_last = -1
    for k,v in state.items():
        if not re.search('^transformer.encoder.layers', k): continue
        k = re.sub('^transformer.encoder.layers.', '', k)
        ind = int(re.search('^\d+', k).group())
        k = re.sub('^\d+\.', '', k)
        ind += 1
        if not ind in res: res[ind] = {}
        res[ind][k] = v
        ind_last = max(ind_last, ind)

    ind_last += 1
    for ind in range(1, ind_last):
        res[ind].update(rotery_states)

    last = {
        ind_last: {
            'output_layer.weight': state['transformer.output_layer.weight'],
        },
    }
    for k,v in state.items():
        if re.search('^transformer.encoder.final_layernorm', k) is None: continue
        k = re.sub('^transformer.encoder.', '', k)
        last[ind_last][k] = v

    res.update(last)
    return res


def convert_chatglm3_6b_pp2hg(pts):
    states = {}
    for ind, pt in pts[1:-1]:
        tmp_state = torch.load(pt, map_location='cpu')
        for k,v in tmp_state.items():
            if re.search('^rotary_pos_emb', k):
                k = f'transformer.{k}'
            else:
                k = f'transformer.encoder.layers.{ind - 1}.{k}'
            states[k] = v

    first_states = torch.load(pts[0][1], map_location='cpu')
    last_states = torch.load(pts[-1][1], map_location='cpu')
    states['transformer.embedding.word_embeddings.weight'] = first_states['word_embeddings.weight']
    states['transformer.output_layer.weight'] = last_states['output_layer.weight']
    for k,v in last_states.items():
        if re.search('^final_layernorm', k) is None: continue
        k = f'transformer.encoder.{k}'
        states[k] = v
    return states


if args.command == 'download':
    model_name = args.model_name
    save_path = args.save_path
    model = AutoModelForCausalLM.from_pretrained(model_name, config=None,
        torch_dtype=torch.half,
        #mirror='tuna',
        #timeout=3600,
    )
    model.save_pretrained(save_path, max_shard_size='1GB')

    if re.search('^decapoda-research/llama', model_name):
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(save_path)


elif args.command == 'hg_to_pp':
    hg_state_path = args.input_path
    pp_state_path = args.save_path
    os.makedirs(pp_state_path, exist_ok=True)

    hg_model = AutoModelForCausalLM.from_pretrained(hg_state_path,
            torch_dtype=torch.half, trust_remote_code=True)
    model_type = hg_model.config.model_type

    state = hg_model.state_dict()
    if re.search('bloom', model_type):
        res = convert_bloom_hg2pp(state)
    elif re.search('llama', model_type):
        res = convert_llama_hg2pp(state)
    elif re.search('baichuan', model_type):
        res = convert_baichuan2_7b_hg2pp(state)
    elif re.search('chatglm', model_type):
        res = convert_chatglm3_6b_hg2pp(state)
    else:
        raise NotImplementedError
    for ind, state in res.items():
        torch.save(state, f'{pp_state_path}/layer_{ind:02d}-model_states.pt')

    hg_model.save_pretrained(pp_state_path, state_dict={})

    if re.search('^decapoda-research/llama', hg_state_path):
        tokenizer = LlamaTokenizer.from_pretrained(hg_state_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hg_state_path, trust_remote_code=True)
    tokenizer.save_pretrained(pp_state_path)
    tokenizer = AutoTokenizer.from_pretrained(pp_state_path, trust_remote_code=True)
    tokenizer.save_pretrained(pp_state_path)
    tokenizer = AutoTokenizer.from_pretrained(pp_state_path, trust_remote_code=True)


elif args.command == 'pp_to_hg':
    pp_state_path = args.input_path
    hg_state_path = args.save_path

    config = AutoConfig.from_pretrained(pp_state_path, trust_remote_code=True)
    model_type = config.model_type

    if hasattr(config, 'num_hidden_layers'):
        n_blocks = config.num_hidden_layers
    else:
        n_blocks = config.num_layers
    pts = []
    for ind in range(n_blocks + 2):
        pt = f'layer_{ind:02d}-model_states.pt'
        pth = osp.join(pp_state_path, pt)
        pts.append((ind, pth))
    pts.sort(key=lambda k: k[0])

    if re.search('bloom', model_type):
        state = convert_bloom_pp2hg(pts)
    elif re.search('llama', model_type):
        state = convert_llama_pp2hg(pts)
    elif re.search('baichuan', model_type):
        state = convert_baichuan2_7b_pp2hg(pts)
    elif re.search('chatglm', model_type):
        state = convert_chatglm3_6b_pp2hg(pts)
    else:
        raise NotImplementedError

    hg_model = AutoModelForCausalLM.from_config(config,
            torch_dtype=torch.half, trust_remote_code=True)
    hg_model.load_state_dict(state)
    hg_model.save_pretrained(hg_state_path, max_shard_size='1GB')

    tokenizer = AutoTokenizer.from_pretrained(pp_state_path, trust_remote_code=True)
    tokenizer.save_pretrained(hg_state_path)
    tokenizer = AutoTokenizer.from_pretrained(hg_state_path, trust_remote_code=True)


