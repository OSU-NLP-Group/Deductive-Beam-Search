import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
from collections import Counter

from src.model import Ranker
from src.openai_api_mp import ChatGPTPool

class Decoder():
    def __init__(self, config) -> None:
        self.device = config.device
        self.prompt = config.prompt
        self.num_sampling = config.num_sampling
        self.num_beams = config.num_beams    

        if config.decode_strategy == "rank":
            self.ranker = Ranker(config)
            # self.ranker.from_pretrained(config.resume_path)
            checkpoint = torch.load(config.resume_path, map_location="cpu")
            checkpoint = checkpoint["model"]
            self.ranker.load_state_dict(checkpoint)
            self.ranker.to(self.device)
            self.decode_fn = self.decode_with_ranking
            self.stop = "\n"
        elif config.decode_strategy == "beam_search":
            self.ranker = Ranker(config)
            # self.ranker.from_pretrained(config.resume_path)
            checkpoint = torch.load(config.resume_path, map_location="cpu")
            checkpoint = checkpoint["model"]
            self.ranker.load_state_dict(checkpoint)
            self.ranker.to(self.device)
            self.decode_fn = self.beam_search
            self.stop = "\n"
        elif config.decode_strategy == "beam_search_with_fact":
            self.ranker = Ranker(config)
            # self.ranker.from_pretrained(config.resume_path)
            checkpoint = torch.load(config.resume_path, map_location="cpu")
            checkpoint = checkpoint["model"]
            self.ranker.load_state_dict(checkpoint)
            self.ranker.to(self.device)
            self.decode_fn = self.beam_search_with_fact
            self.stop = "\n"
        else:
            self.decode_fn = self.decode_direct
            self.stop = None
        
        if "gpt" in config.decoder_path:
            self.model = ChatGPTPool(
                num_sampling = config.num_sampling,
                temperature = 1.0,
                max_tokens = 256,
                stop = self.stop,
            )
        else:
            if config.is_greedy:
                self.sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=512)
            else:
                self.sampling_params = SamplingParams(n=self.num_sampling, temperature=1, top_p=0.95, max_tokens=512, stop=self.stop)
                self.model = LLM(model=config.decoder_path, seed=42, max_num_batched_tokens=4096, tensor_parallel_size=config.num_gpus_decode, gpu_memory_utilization=0.85) # , dtype="float32" , tensor_parallel_size=2
            
    # TODO: batch inference
    def decode(self, question):
        return self.decode_fn(question)
    
    @torch.inference_mode()
    def decode_with_ranking(self, question):
        raise NotImplementedError
            
    @torch.inference_mode()
    def decode_direct(self, question):
        raise NotImplementedError

    @torch.inference_mode()
    def beam_search(self, question):
        raise NotImplementedError
    
    @torch.inference_mode()
    def beam_search_with_fact(self, question):
        raise NotImplementedError
    
class FastDecoder(Decoder):
    def __init__(self, config) -> None:
        super().__init__(config)

    @torch.inference_mode()
    def decode_with_ranking(self, question):
        answers = []
        context = "\nQuestion:\n" + question +  "\nAnswer:\n"
        terminated = False
        step = 0
        rationales = []
        while not terminated and step < 20:
            outputs = self.model.generate(self.prompt + context, self.sampling_params, use_tqdm=False)[0].outputs
            answers = []
            for output in outputs:
                answers.append(output.text.strip().split("\n")[0].strip())
            contexts = [context] * len(answers)
            scores = self.ranker(contexts, answers).cpu().squeeze()
            sorted, indices = torch.sort(scores, descending=True)
            rationale = answers[indices[0]]
            rationales.append(rationale)
            context += f"{rationale}\n"
            if "Final Answer" in rationale:
                terminated = True
            step += 1
        return rationales
            
    @torch.inference_mode()
    def decode_direct(self, question):
        context = "\nQuestion:\n" + question +  "\nAnswer:\n"
        outputs = self.model.generate(self.prompt + context, self.sampling_params, use_tqdm=False)[0].outputs
        answers = []
        for output in outputs:
            answers.append(output.text.strip())
        return answers
    
    @torch.inference_mode()
    def beam_search(self, question):
        self.sampling_params = SamplingParams(n=self.num_beams, temperature=1, top_p=0.95, max_tokens=512)
        answers = []
        context = "\nQuestion:\n" + question +  "\nAnswer:\n"
        global_terminated = False
        step = 0
        beams = [(context, 1, False)] * self.num_beams # (current rationale, score, terminated)
        completed_rationales = []
        while not global_terminated and step < 20:
            current_beams = []
            for beam, score, terminated in beams:
                # if terminated, leave it alone
                if terminated:
                    current_beams.append((beam, score, terminated))
                    continue
                    
                # otherwise, generate next rationale
                outputs = self.model.generate(self.prompt + beam, self.sampling_params, use_tqdm=False)[0].outputs
                answers = []
                for output in outputs:
                    answers.append(output.text.strip().split("\n")[0].strip())
                contexts_for_ranker = [beam] * len(answers)
                scores = self.ranker(contexts_for_ranker, answers).cpu().squeeze()
                sorted_scores, indices = torch.sort(scores, descending=True)
                
                # calculate current score
                for _ in range(self.num_beams):
                    current_beam = beam + answers[indices[_]] + "\n"
                    current_score = score * scores[indices[_]]
                    # if termintated, add to completed rationales
                    if "Final Answer" in answers[indices[_]] or "Final answer" in answers[indices[_]]:
                        terminated = True
                        completed_rationales.append((current_beam, current_score.item()))
                    current_beams.append((current_beam, current_score.item(), terminated))
            sorted_beams = sorted(current_beams, key=lambda x: x[1], reverse=True)
            beams = sorted_beams[:self.num_beams]
            flag = True
            for _ , _, terminated in beams:
                if not terminated:
                    flag = False
                break
            global_terminated = flag
            step += 1
            
        return beams, completed_rationales
    
    @torch.inference_mode()
    def beam_search_with_fact(self, question):
        answers = []
        context = "\nQuestion:\n" + question +  "\nAnswer:\n"
        if_verify = False
        step = 0
        while not if_verify and step < 10:
            fact_sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=512, stop="\n")
            outputs = self.model.generate(self.prompt + context, fact_sampling_params, use_tqdm=False)[0].outputs
            context += outputs[0].text
            context += "\n"
            if "Reasoning" in outputs[0].text:
                if_verify = True
            step += 1
        
        global_terminated = False
        beams = [(context, 1, False)] * self.num_beams # (current rationale, score, terminated)
        completed_rationales = []
        
        while not global_terminated and step < 10:
            current_beams = []
            for beam, score, terminated in beams:
                # if terminated, leave it alone
                if terminated:
                    current_beams.append((beam, score, terminated))
                    continue
                    
                # otherwise, generate next rationale
                outputs = self.model.generate(self.prompt + beam, self.sampling_params, use_tqdm=False)[0].outputs
                answers = []
                for output in outputs:
                    answers.append(output.text.strip().split("\n")[0].strip())
                contexts_for_ranker = [beam] * len(answers)
                scores = self.ranker(contexts_for_ranker, answers).cpu().squeeze()
                sorted_scores, indices = torch.sort(scores, descending=True)

                # calculate current score
                for _ in range(self.num_beams):
                    current_beam = beam + answers[indices[_]] + "\n"
                    current_score = score * scores[indices[_]]
                    # if termintated, add to completed rationales
                    if "Final Answer" in answers[indices[_]] or "Final answer" in answers[indices[_]]:
                        terminated = True
                        completed_rationales.append((current_beam, current_score.item()))
                    current_beams.append((current_beam, current_score.item(), terminated))
                        
            sorted_beams = sorted(current_beams, key=lambda x: x[1], reverse=True)
            beams = sorted_beams[:self.num_beams]
            flag = True
            for _ , _, terminated in beams:
                if not terminated:
                    flag = False
                break
            global_terminated = flag
            step += 1
            
        return beams, completed_rationales
    
class ChatGPTDecoder(Decoder):
    def __init__(self, config) -> None:
        super().__init__(config)
            
    @torch.inference_mode()
    def decode_direct(self, question):
        context = "\nQuestion:\n" + question +  "\nAnswer:\n"
        outputs = self.model.chat_single_round(self.prompt + context)
        answers = []
        for output in outputs:
            answers.append(output.strip())
        return answers
    
    @torch.inference_mode()
    def beam_search(self, question):
        answers = []
        context = "\nQuestion:\n" + question +  "\nAnswer:\n"
        global_terminated = False
        step = 0
        beams = [(context, 1, False)] * self.num_beams # (current rationale, score, terminated)
        completed_rationales = []
        while not global_terminated and step < 50:
            current_beams = []
            for beam, score, terminated in beams:
                # if terminated, leave it alone
                if terminated:
                    current_beams.append((beam, score, terminated))
                    continue
                    
                # otherwise, generate next rationale
                outputs = self.model.chat_single_round(self.prompt + beam)
                answers = []
                for output in outputs:
                    answer = output.strip().split("\n")[0].strip()
                    if len(answer) < 1:
                        continue
                    answers.append(answer)
                contexts_for_ranker = [beam] * len(answers)
                scores = self.ranker(contexts_for_ranker, answers).cpu().squeeze()
                sorted_scores, indices = torch.sort(scores, descending=True)
                
                # calculate current score
                for _ in range(min(len(answers), self.num_beams)):
                    current_beam = beam + answers[indices[_]] + "\n"
                    current_score = score * scores[indices[_]]
                    # if termintated, add to completed rationales
                    if "Final Answer" in answers[indices[_]] or "Final answer" in answers[indices[_]]:
                        terminated = True
                        completed_rationales.append((current_beam, current_score.item()))
                    current_beams.append((current_beam, current_score.item(), terminated))
            sorted_beams = sorted(current_beams, key=lambda x: x[1], reverse=True)
            beams = sorted_beams[:self.num_beams]
            flag = True
            for _ , _, terminated in beams:
                if not terminated:
                    flag = False
                break
            global_terminated = flag
            step += 1 
        return (beams, completed_rationales)
    
    @torch.inference_mode()
    def beam_search_with_fact(self, question):
        answers = []
        context = "\nQuestion:\n" + question +  "\nAnswer:\n"
        if_verify = False
        step = 0
        self.model.num_sampling = 1
        while not if_verify and step < 10:
            self.model.num_sampling = 1
            outputs = self.model.chat_single_round(self.prompt + context)
            context += outputs[0]
            context += "\n"
            if "Reasoning" in outputs[0]:
                if_verify = True
                
        self.model.num_sampling = self.num_sampling
        global_terminated = False
        beams = [(context, 1, False)] * self.num_beams # (current rationale, score, terminated)
        completed_rationales = []
        while not global_terminated and step < 20:
            current_beams = []
            for beam, score, terminated in beams:
                # if terminated, leave it alone
                if terminated:
                    current_beams.append((beam, score, terminated))
                    continue
                    
                # otherwise, generate next rationale
                outputs = self.model.chat_single_round(self.prompt + beam)
                answers = []
                for output in outputs:
                    answers.append(output.strip().split("\n")[0].strip())
                contexts_for_ranker = [beam] * len(answers)
                scores = self.ranker(contexts_for_ranker, answers).cpu().squeeze()
                sorted_scores, indices = torch.sort(torch.tensor(scores), descending=True)

                # calculate current score
                for _ in range(self.num_beams):
                    current_beam = beam + answers[indices[_]] + "\n"
                    current_score = score * scores[indices[_]]
                    # if termintated, add to completed rationales
                    if "Final Answer" in answers[indices[_]] or "Final answer" in answers[indices[_]]:
                        terminated = True
                        completed_rationales.append((current_beam, current_score.item()))
                    current_beams.append((current_beam, current_score.item(), terminated))
            sorted_beams = sorted(current_beams, key=lambda x: x[1], reverse=True)
            beams = sorted_beams[:self.num_beams]
            flag = True
            for _ , _, terminated in beams:
                if not terminated:
                    flag = False
                break
            global_terminated = flag
            step += 1
        return (beams, completed_rationales)