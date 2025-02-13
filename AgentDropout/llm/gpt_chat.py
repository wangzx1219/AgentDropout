import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import async_timeout
from transformers import AutoTokenizer

from AgentPrune.llm.format import Message
from AgentPrune.llm.price import cost_count, cost_count_llama3, cost_count_deepseek
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry


load_dotenv()
MINE_BASE_URL = ""
MINE_API_KEYS = ""

# print(MINE_BASE_URL)


# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
# async def achat(
#     model: str,
#     msg: List[Dict],):
#     request_url = MINE_BASE_URL
#     authorization_key = MINE_API_KEYS
#     headers = {
#         'Content-Type': 'application/json',
#         'authorization': authorization_key
#     }
#     data = {
#         "name": model,
#         "inputs": {
#             "stream": False,
#             "msg": repr(msg),
#         }
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(request_url, headers=headers ,json=data) as response:
#             response_data = await response.json()
#             if isinstance(response_data['data'],str):
#                 prompt = "".join([item['content'] for item in msg])
#                 cost_count(prompt,response_data['data'],model)
#                 return response_data['data']
#             else:
#                 raise Exception("api error")

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict],):
    api_kwargs = dict(api_key = MINE_API_KEYS, base_url = MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count(prompt, response_message, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(6))
async def achat_deepseek(model: str, msg: List[Dict],):
    model = ''
    # print(1111111)
    api_kwargs = dict(api_key = deepseek_api, base_url = deepseek_url)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg,temperature=1.0,top_p=1.0)
        # print(completion)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_deepseek(prompt, response_message, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def achat_llama(model: str, msg: List[Dict]):
    # print(111111111111)
    api_kwargs = dict(api_key = "token-abc123", base_url = "http://localhost:6305/v1")
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_llama3(prompt, response_message, model)
            return response_message

    except Exception as e:
        # 记录日志或处理异常
        print(f"Error in achat_llama: {e}")
        # raise
    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('deepseek')
class DeepseekChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_deepseek(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('llama')
class LlamaChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name
        # print(11111111111111111111)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_llama(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass