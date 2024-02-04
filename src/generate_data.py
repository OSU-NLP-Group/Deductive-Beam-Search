import re
import json
import torch
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationConfig
from vllm import LLM, SamplingParams

from src.decoder import Decoder
from src.parser_utils import get_parser
from prompts.gsm8k import prompt
from src.model import Ranker

def generate_correct_rationales():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to(torch.device("cuda:0"))
    datasets = load_dataset("gsm8k", "main")
    dataset = datasets["train"]
    pattern = r"<<(.*?)>>"
    datas = []
    prompt = "Please rephrase the following sentence and keep the equation between <<>> unchanged. For exaxmple:\n{example}"
    # model = ChatGPTAPI(prompt)
    count_done = 0
    with open("data/gsm8k/gsm8k.txt", "r", encoding="utf-8") as fin:
        for _ in fin.readlines():
            count_done += 1
    sid = 0
    for sample in tqdm(dataset):
        if sid < count_done:
            sid += 1
            continue
        # model.reset()
        question = sample["question"]
        gold_answer = sample["answer"]
        final_answer = gold_answer.split("####")[-1].strip()
        # equations = re.findall(pattern, gold_answer)
        rationales = gold_answer.split("####")[0].split("\n")[:-1]
        diverse_rationales = []
        for r in rationales:
            model.reset()
            # print(r)
            rs = [r]
            state = f"{prompt}\n{r}"
            for _ in range(4):
                answer = model.chat_multi_round(state)
                # print(answer)
                rs.append(answer)
                state = None
            diverse_rationales.append(rs)
        write_data = {
            "question": question,
            "rationales": diverse_rationales,
            "chain_length": len(rationales),
            "final_answer": final_answer,
        }
        with open("data/gsm8k/gsm8k.txt", "a+", encoding="utf-8") as fout:
            fout.write(f"{json.dumps(write_data)}\n")

def confuse(rationales):
    """
        Generate false samples for given rationales
    """
    
    pattern = r"([\$A-Za-z0-9\%\.]+\s*[\+\-\*\/x])+\s*[\$A-Za-z0-9\%\.]+\s*(=\s*[\$A-Za-z0-9\%\.]+)"
    number_set = set()
    for rs in rationales:
        rationale = rs[0]
        if rationale.endswith("."):
            rationale = rationale[:-1]
        equation = re.search(pattern, rationale)
        if equation is None:
            continue
        equation = equation.group()
        eq = equation.split("=")[0].strip()
        elements = re.split(r"[\+\-\/\*x]", eq)
        for e in elements:
            number_set.add(e.strip())
        number_set.add(equation.split("=")[-1].strip())
    number_set = list(number_set)
    print(number_set)
    false_rationales = {}
    for rid, rs in enumerate(rationales):
        f_rs = []
        for r in rs:
            print(r)
            if r.endswith("."):
                r = r[:-1]
            equation = re.search(pattern, r)
            # get some following rationales
            following_rationales = []
            for fid, tmp_rs in enumerate(rationales):
                if fid <= rid:
                    continue
                following_rationales.extend(tmp_rs)
            print(len(following_rationales))
            if equation is None: # if there is no eqaution, continue
                continue
            # if there is an equation, replace some elements in it
            e_start = equation.start()
            e_end = equation.end()
            equation = equation.group()
            eq = equation.split("=")[0].strip()
            symbols = re.findall(r"[\+\-\/\*x]", eq)
            elements = re.split(r"[\+\-\/\*x]", eq)
            elements = [ele.strip() for ele in elements]
            count = 0
            while count < 10:
                try:
                    replaced_index = random.choice(range(len(elements)))
                    print(elements[replaced_index])
                    tmp_number_set = [n for n in number_set]
                    tmp_number_set.remove(elements[replaced_index])
                    elements[replaced_index] = random.choice(tmp_number_set)
                    false_eq = f"{elements[0]}"
                    for sid, symbol in enumerate(symbols):
                        if symbol == "x":
                            symbol = "*"
                        false_eq += f"{symbol}{elements[sid + 1]}"
                    try:
                        print(false_eq)
                        value = eval(false_eq)
                        f_r = r[:e_start - 1] + f" {false_eq} = {value} " + r[e_end:]
                        f_rs.append(f_r)
                    except:
                        print(false_eq)
                        value = equation.split("=")[-1].strip()
                        f_r = r[:e_start - 1] + f" {false_eq} = {value} " + r[e_end:]
                        f_rs.append(f_r)
                    break
                except:
                    count += 1
                    continue
        false_rationales[rid] = f_rs
    return false_rationales        
            
def generate_ranking_negative_samples():
    """
    1. gold answer
    2. gold answer but change one number
    2. similar answer that change one number
    3. other rationales from different question
    """
    def get_some_random_samples(excluded_id):
        id_list = list(range(len(id2false_rationales)))
        id_list.remove(excluded_id)
        id_list = random.sample(id_list, 1)
        selected_false_rationales = []
        for sid in id_list:
            false_rationales = None
            while false_rationales is None or len(false_rationales) == 0:
                false_rationales = id2false_rationales.get(sid)
                sid += 1
            s_false_rationales = []
            for v in false_rationales.values():
                s_false_rationales.extend(v)
            selected_false_rationales.append(random.choice(s_false_rationales))
        return selected_false_rationales
            
            
    id2false_rationales = {}
    with open("data/gsm8k/false_rationales.txt", "r") as fin:
        for line in fin.readlines():
            data = json.loads(line.strip())
            id2false_rationales.update({data["id"]: data["false_rationales"]})
    samples = {}
    with open("data/gsm8k/gsm8k_clean.txt", "r") as fin:
        for line in fin.readlines():
            data = json.loads(line.strip())
            metadata = {**data}
            samples.update({data["id"]: metadata})
    with open("data/gsm8k/train_short.json", "r") as fin:
        shortened_contexts = json.load(fin)
    answer2context = {}
    for sample in shortened_contexts:
        answer2context.update({sample["answer"]: sample["context"]})
    datasets = load_dataset("gsm8k", "main")
    dataset = datasets["train"]
    write_datas = []
    for sid, sample in enumerate(dataset):
        false_rationales = id2false_rationales[sid]
        correct_metadata = samples[sid]
        correct_rationales = correct_metadata["rationales"]
        chain_length = correct_metadata["chain_length"]
        context = correct_metadata["question"]
        for i in range(chain_length):
            i_correct_rationales = correct_rationales[i]
            i_false_rationales = false_rationales.get(str(i))
            if i_false_rationales is None or len(i_false_rationales) <= 1:
                continue
            cr = i_correct_rationales[0]
            answers = {
                0: cr,
                1: i_false_rationales[0],
                2: random.choice(i_false_rationales[1:]),
                3: get_some_random_samples(sid)[0]
            }
            write_datas.append({
                "context": context,
                "answers": answers,
            })
            context += f" {cr}"
    print(len(write_datas)) # 111280
    with open("data/gsm8k/train_ranking_small_full_multi_negative.json", "w") as fout:
        json.dump(write_datas, fout, indent=1, ensure_ascii=False)


@torch.inference_mode()
def generate_hard_negative():
    torch.manual_seed(42)
    random.seed(42)

    parser = get_parser()
    config = parser.parse_args()

    config.device = torch.device("cuda")
    config.generation_config = GenerationConfig(
        do_sample=True,
        temperature=1,
        top_p=0.95,
        max_new_tokens=512,
    )
    config.prompt = prompt
    
    sampling_params = SamplingParams(n=10, temperature=1, top_p=0.95, max_tokens=512)

    # decoder = Decoder(config)
    
    dataset = load_dataset("data/cache/meta-math___json/meta-math--MetaMathQA-b6af0a8ce3115a0e")["train"]
    datas = []
    all_rationales = []
    for sample in dataset:
        if sample["type"].startswith("GSM"):
            response = sample["response"].split("\n")[:-1]
            final_rationale = response[-1].replace("####", "").strip()
            response[-1] = f"Final Answer: {final_rationale}"
            sample["response"] = response            
            datas.append(sample)
            all_rationales.extend(response)
        
    llm = LLM(model="~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235/", seed=42) # , dtype="float32"
    ranker = Ranker(config)
    checkpoint = torch.load("ckpts/gsm8k/gsm8k_ranking_multi_neg_margin.pt", map_location="cpu")
    checkpoint = checkpoint["model"]
    del(checkpoint["model.embeddings.position_ids"])
    ranker.load_state_dict(checkpoint)
    ranker.cuda()
    ranker.eval()

    progressbar = tqdm(range(len(datas)))
    for did, data in enumerate(datas):
        question = data["query"]
        answer = data["response"]
        context = "\nQuestion:\n" + question + "\nAnswer:\n"
        for _ in range(len(answer)):
            outputs = llm.generate(prompt + context, sampling_params, use_tqdm=False)[0].outputs
            generated_answers = []
            for output in outputs:
                generated_answers.append(output.text.strip().split("\n")[0].strip())
            contexts = [context] * len(generated_answers)
            scores = ranker(contexts, generated_answers).cpu().squeeze()
            # scores = torch.tensor(scores).squeeze()
            sorted, indices = torch.sort(scores, descending=True)
            # print(answers)
            # input()
            write_data = {
                "sid": did,
                "context": context.strip(),
                "answers": {
                    "0": answer[_],
                    "1": generated_answers[indices[-1]],
                    "2": random.choice(all_rationales),
                }
            }
            with open("data/gsm8k/hard_negative_metamathqa.txt", "a+") as fout:
                fout.write(f"{json.dumps(write_data)}\n")
            context += answer[_] + "\n"
            # print(context)
            # input()
        progressbar.update(1)

if __name__ == "__main__":
    generate_hard_negative()