
import os
import re

import torch
import deepspeed
from transformers import StoppingCriteria
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import pipeline

torch.set_grad_enabled(False)

'''
    deepspeed的方式，每个gpu上有一个进程，每个进程都加载一遍完整的模型，容易导致oom
    运行:
        deepspeed --num_gpus 4 --num_nodes 1 demo.py
'''


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[2277, 29937]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


def create_tokenizer(model_name):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if re.search('llama', model_name):
        tokenizer = LlamaTokenizer.from_pretrained(model_name, config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, trust_remote_code=True)

    return tokenizer


def create_model(model_name):
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.cuda()
    model.eval()

    #  infer_config = dict(
    #          tensor_parallel={'tp_size': world_size},
    #          dtype=torch.half,
    #          replace_with_kernel_inject=True,
    #  )
    #  ## 使用pipeline
    #  model = pipeline('text-generation', model=model_name,
    #          device=local_rank,
    #          torch_dtype=torch.half,
    #          tokenizer=tokenizer,
    #  )
    #  model.model = deepspeed.init_inference(model.model, config=infer_config)

    return model


@torch.inference_mode()
def generate(model, tokenizer, prompt):
    #  res = model([prompt,], do_sample=False, temperature=0.7, max_new_tokens=300)
    stop_crit = [EosListStoppingCriteria(
        tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)), ]

    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to("cuda")
    output_ids = model.generate(inputs.input_ids, do_sample=True,
            temperature=0.7, max_new_tokens=300, stopping_criteria=stop_crit)
    res = tokenizer.decode(output_ids[0])
    return res, output_ids



def infer_instruct(model):
    model = create_model(model_name)
    tokenizer = create_tokenizer(model_name)

    prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    简化以下段落，使其更易理解

    ### Input:
    尽管人们普遍认为互联网使我们能够与世界各地的人联系，但仍有一些人不熟悉其基本功能，不理解为什么它变得如此普遍，或者它的真正能力是什么。

    ### Response:'''

    res = generate(model, tokenizer, prompt)
    print(res)


def infer_tool(model):
    model = create_model(model_name)
    tokenizer = create_tokenizer(model_name)

    prompt_header = '''Answer the following questions as best as you can. You have access to the following tool(s):\n'''
    prompt_api_desc = '''getTeamRoster: Retrieve a list of current players on an MLB team.\nParameters: {\"teamId\": \"Required. integer. The ID of the team whose roster is being requested.\"}\nOutput: An array of player objects, each containing the player's ID, name, position, and other relevant information.\n - Format: application/json\n - Structure: Array[Object{id, name, position, otherInfo}]\ngetGameBoxscore: Get the box score for a specific MLB game.\nParameters: {\"gameId\": \"Required. integer. The ID of the game whose box score is being requested.\"}\nOutput: An object containing detailed information about the game, including the final score, player statistics, and other relevant data.\n - Format: application/json\n - Structure: Object{finalScore, playerStats: Array[Object{id, name, statistic}], otherData}\ngetPlayerStats: Access statistics on a player: batting average, ERA, etc.\nParameters: {\"playerId\": \"Required. integer. The ID of the player whose statistics are being requested.\"}\nOutput: An object containing the player's statistics for the current season, including batting average, home runs, RBIs, ERA, and other relevant data.\n - Format: application/json\n - Structure: Object{battingAverage, homeRuns, RBIs, ERA, otherStats}\ngetTeamStandings: Retrieve team standings and other league-wide statistics.\nParameters: {\"season\": \"integer. The year of the season for which standings are being requested. Defaults to the current season.\"}\nOutput: An array of team objects, each containing the team's ID, name, win-loss record, and other relevant information. Additionally, league-wide statistics such as runs scored, home runs, and batting average are included.\n - Format: application/json\n - Structure: Array[Object{id, name, winLossRecord, otherStats}]\n'''
    prompt_ask = '''### Human: I want to know the number of RBIs for Fernando Tatis Jr. this season. \n'''
    prompt_ans = '''### Assistant: '''

    prompt = prompt_header + prompt_api_desc + prompt_ask + prompt_ans
    res = generate(model, tokenizer, prompt)[0]

    #  out = re.sub('^{prompt}', '', res)
    #  if not re.search('^#### api call', out) is None:
    #      prompt_api_res = '''\n#### api output\nParameter type error: ...\n'''
    #      prompt = res + prompt_api_res + prompt_ans

    prompt_api_res = '''\n#### api output\nParameter type error: \"teamId\", expected integer, but got string. You need to change the input and try again.\n'''
    prompt = res + prompt_api_res + prompt_ans
    res = generate(model, tokenizer, prompt)[0]
    print(res)


if __name__ == '__main__':

    model_name = 'decapoda-research/llama-7b-hf'
    #  model_name = 'bigscience/bloomz-560m'
    model_name = 'checkpoints/tool_alpaca'

    #  infer_instruct(model_name)
    infer_tool(model_name)


