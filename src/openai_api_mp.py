import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import openai
import tiktoken
from openai.error import InvalidRequestError
from typing import List, Dict, AnyStr
from tenacity import stop_after_attempt, retry, wait_random_exponential, retry_if_not_exception_type
from multiprocessing import Pool, Value

def openai_setup(api_key = None):
    if api_key:
        openai.api_key = api_key
    else:
        openai.api_key = os.environ['openai_api_key']

class ChatGPTPool:
    def __init__(
            self,
            api_key = "sk-your_api_key",
            prompt: AnyStr = None,
            history_messages: List[Dict[AnyStr, AnyStr]] = None,
            num_sampling = 1,
            temperature = 1.0,
            max_tokens = 256,
            stop = "\n",
    ) -> None:
        openai_setup(api_key=api_key)
        if history_messages:
            self.messages = history_messages
        else:
            self.messages = []
            if prompt:
                msg = {"role": "system","content": prompt}
                self.messages.append(msg)
        
        self.money = 0
        self.num_sampling = num_sampling
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.prompt = prompt
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")
        
    def check_cost(self):
        return self.money

    def chat_single_round(
        self,
        message,
    ):
        num_tokens = 0
        messages = []
        if self.prompt:
            messages.append({"role": "system", "content": self.prompt})
            num_tokens += len(self.encoding.encode(self.prompt))
        e = {"role": "user", "content": message}
        messages.append(e)
        num_tokens += len(self.encoding.encode(message))
        self.money += 0.0015 / 1000 * num_tokens * self.num_sampling
        
        with Pool(self.num_sampling) as p:
            contents = p.map(chat_single_round, [(messages, self.temperature, self.max_tokens, self.stop)] * self.num_sampling)
        
        num_tokens = sum([len(self.encoding.encode(content)) for content in contents])
        self.money += 0.002 / 1000 * num_tokens
        return contents


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=60), retry=retry_if_not_exception_type(InvalidRequestError))
def chat_single_round(
    messages,
):
    temperature = messages[1]
    max_tokens = messages[2]
    stop = messages[3]
    messages = messages[0]

    prompt = []
    for message in messages:
        prompt.append(message["content"])
    prompt = "\n".join(prompt)
    completion = openai.Completion.create(
        model = 'gpt-3.5-turbo-instruct',
        prompt = prompt,
        temperature = temperature,
        max_tokens = max_tokens,
        stop = stop,
    )
    content = completion.choices[0].text
    return content
