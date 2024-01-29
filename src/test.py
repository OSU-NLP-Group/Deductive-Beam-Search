import os
import json
import torch
import random
torch.manual_seed(42)
random.seed(42)

from datasets import load_dataset

from src.decoder import FastDecoder, ChatGPTDecoder
from src.parser_utils import get_parser
from prompts.gsm8k import prompt as math_prompt
from prompts.strategyqa import prompt as cs_prompt
from prompts.csqa import prompt as csqa_prompt
from prompts.coin import prompt as coin_prompt
from prompts.strategyqa import recall_prompt as cs_recall_prompt
from prompts.csqa import recall_prompt as csqa_recall_prompt
from tqdm import tqdm

MATH_DATASET = [
    "gsm8k",
    "svamp",
    "aqua",
    "singleeq",
    "multiarith",
]
COMMONSENSE_DATASET = [
    "strategyqa",
    "csqa",
]
SYMBOLIC_DATASET = [
    "coin",
]

MODEL_NAME = ["7b", "13b", "70b", "gpt"]


def load_dataset_from_config(config):
    """
        The returned datas should be a list and each element should be a dict containing keyword "quesiton".
        [
            {
                "id": id,
                "question": question,
            }
        ]
    """
    datas = []
    if config.test_dataset == "gsm8k":
        dataset = load_dataset("data/cache/gsm8k", "main")["test"]
        for did, data in enumerate(dataset):
            datas.append({
                "id": did,
                "question": data["question"],
            })
    elif config.test_dataset == "svamp":
        dataset = load_dataset("data/cache/ChilleD___json/ChilleD--SVAMP-4bd8179a65d5f05b")["test"]
        for data in dataset:
            datas.append({
                "id": data["ID"],
                "question": data["Body"] + " " + data["Question"],
            })
    elif config.test_dataset == "aqua":
        dataset = []
        with open("data/cache/AQuA/test.json", "r") as fin:
            for line in fin.readlines():
                dataset.append(json.loads(line.strip()))
            # dataset = json.load(fin)
        for did, data in enumerate(dataset):
            datas.append({
                "id": did,
                "question": data["question"],
            })
    elif config.test_dataset == "addsub":
        dataset = load_dataset("data/cache/allenai___lila", "addsub")["test"]
        for did, data in enumerate(dataset):
            datas.append({
                "id": did,
                "question": data["input"],
            })
    elif config.test_dataset == "singleeq":
        data_dir = "data/cache/SingleEq"
        for i in range(5):
            with open(os.path.join(data_dir, f"test{i}")) as fin:
                for lid, line in enumerate(fin.readlines()):
                    if lid % 3 == 0:
                        datas.append({
                            "id": int(lid / 3),
                            "question": line.strip(),
                        })
    elif config.test_dataset == "multiarith":
        dataset = load_dataset("data/cache/ChilleD___json/ChilleD--MultiArith-2e3d95e2a4ce9083")["test"]
        for did, data in enumerate(dataset):
            datas.append({
                "id": did,
                "question": data["question"],
            })
    elif config.test_dataset == "strategyqa":
        with open("data/cache/strategyqa/test_set.json", "r") as fin:
            dataset = json.load(fin)
        for data in dataset:
            datas.append({
                "id": data["qid"],
                "question": data["question"],
            })
    elif config.test_dataset == "csqa":
        dataset = load_dataset("data/cache/commonsense_qa")["validation"]
        for data in dataset:
            labels = data["choices"]["label"]
            texts = data["choices"]["text"]
            choices = []
            for label, text in zip(labels, texts):
                choices.append(f"{label}. {text}")
            choices = "\n".join(choices)
            question = data["question"] + "\n" + choices
            datas.append({
                "id": data["id"],
                "question": question,
            })
    elif config.test_dataset == "coin":
        dataset = load_dataset("data/cache/skrishna___json/skrishna--coin_flip-8305ab6800b027bf")["test"]
        for did, data in enumerate(dataset):
            question = data["inputs"].replace("Q:", "").strip()
            datas.append({
                "id": did,
                "question": question,
            })
    else:
        print(f"[WARNING] {config.test_dataset} is not an option!")
        raise NotImplementedError
    return datas

def load_prompt_from_config(config):
    if config.test_dataset in MATH_DATASET:
        prompt = math_prompt
    elif config.test_dataset in COMMONSENSE_DATASET:
        if config.test_dataset == "csqa":
            if "fact" in config.decode_strategy:
                prompt = csqa_recall_prompt
            else:
                prompt = csqa_prompt
        else:
            if "fact" in config.decode_strategy:
                prompt = cs_recall_prompt
            else:
                prompt = cs_prompt
    elif config.test_dataset in SYMBOLIC_DATASET:
        if config.test_dataset == "coin":
            prompt = coin_prompt
    else:
        print(f"[WARNING] {config.test_dataset} is not an option!")
        raise NotImplementedError
    return prompt

def load_dataset_and_prompt(config):
    return load_dataset_from_config(config), load_prompt_from_config(config)
        

@torch.inference_mode()
def main():
    torch.manual_seed(42)
    random.seed(42)

    parser = get_parser()
    config = parser.parse_args()
    
    config.device = torch.device("cuda")
    
    # load dataset, prompt, and decoder
    dataset, prompt = load_dataset_and_prompt(config)
    config.prompt = prompt
    if "gpt" in config.decoder_path:
        decoder = ChatGPTDecoder(config)
    else:
        decoder = FastDecoder(config)
    
    # load from previous infer checkpoint
    if config.resume_path is not None:
        ckpt_name = config.resume_path.split(".")[0].split("/")[-1]
    else:
        ckpt_name = "None"
    
    for name in MODEL_NAME:
        if name in config.decoder_path:
            model_abbr_name = name
        else:
            if "/" in config.decoder_path:
                model_abbr_name = config.decoder_path.split("/")[-1]
            else:
                model_abbr_name = config.decoder_path
    
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    config.output_dir = os.path.join(config.output_dir, model_abbr_name)
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    save_path = os.path.join(config.output_dir, f"{config.test_dataset}_{model_abbr_name}_{ckpt_name}_{config.decode_strategy}_N{config.num_sampling}_B{config.num_beams}.txt")
    if os.path.exists(save_path):
        with open(save_path, "r") as fin:
            line = fin.readlines()[-1]
            data = json.loads(line.strip())
            for key in data.keys():
                continue
            last_sid = int(key)
    else:
        last_sid = -1

    # inference
    progressbar = tqdm(range(len(dataset)))
    for sid, sample in enumerate(dataset):
        if sid <= last_sid:
            progressbar.update(1)
            continue
        question = sample["question"]
        answers = decoder.decode_fn(question)
        
        # save inference checkpoint
        if "beam_search" in config.decode_strategy:
            beams, full_rationales = answers
            with open(save_path, "a+") as fout:
                fout.write(f"{json.dumps({sid: beams[0]})}\n")
            with open(f"{save_path}.full", "a+") as fout:
                fout.write(f"{json.dumps({sid: full_rationales})}\n")
        else:
            with open(save_path, "a+") as fout:
                fout.write(f"{json.dumps({sid: answers})}\n")
        progressbar.update(1)
        
if __name__ == "__main__":
    main()