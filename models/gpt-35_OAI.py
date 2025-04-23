from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
import helper_functions

load_dotenv()

"""

Model: GPT 3.5 hosted by Open AI, see implementation for details on version
Authors: Breck Baldwin.
How to setup model to run: Model requires Open AI authentication and endpoints as shown in code. Easiest way to set variables is to have a `.env` file in your home directory for `dotenv` to find. DO NOT PUT `.env` in repo or 
you risk putting credentials in the repo. 



 * Copyright 2024 Comcast Cable Communications Management, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

MODEL = OpenAI(
                api_key=os.environ["OPEN_AI_KEY"]
            )
MODEL_NAME = "gpt-3.5-turbo"

seen_before = {} #cache previous results for run to avoid variation

def run(prompt: list, config: dict) -> (str):
    """
    Runs a prompt and returns the content portion of the response and 
    return run parameters.
    Arguments:
       prompt: list of dicts, example: [{"role": "user", "content": question}]
       config: dict, {'temperature': float probability,
                      'seed': int
                      'top_p_k': float probability,
                      'rewrite_inst': str
                      }
    Returns:
       str: payload
       dict: run configuration used--may have rewritten prompt or other mods
    """
    cache_used = False
    logprobs_config = config.get('logprobs', False)
    if config.get('rewrite_inst', None) is not None:
        prompt, cache_used = \
            helper_functions.apply_rewrite(prompt, config, seen_before, MODEL, 
                                            MODEL_NAME)
    if config.get('prefix', None) is not None:
        prompt = [{'role': 'user',
                   'content': config['prefix'] + prompt[0]['content']}]
    if config.get('suffix', None) is not None:
        prompt = [{'role': 'user',
                   'content': prompt[0]['content'] + config['suffix']}]
    
    model = MODEL_NAME
    if config.get('even_model', None) is not None \
        and config['rubric_counter'] % 2 == 0:
        model = config['even_model']
    elif config.get('odd_model', None) is not None \
        and config['rubric_counter'] % 2 == 1:
        model=config['odd_model']
        
    
    response = MODEL.chat.completions.create(messages=prompt, model=model)
    
    response = MODEL.chat.completions.create(
                    messages=prompt,
                    model=model,
                    temperature=config['temperature'],
                    seed=config['seed'],
                    top_p=config['top_p_k'],
                    logprobs=logprobs_config
                )
    if logprobs_config:
        logprobs = response.choices[0].logprobs.content
    else:
        logprobs = []
    return (response.choices[0].message.content,
            {
            'prompt':prompt, 
            'model_name': model, 
            'temperature': config['temperature'],
            'seed': config['seed'],
            'top_p_k': config['top_p_k'],
            'rewrite_inst': config.get('rewrite_inst', None),
            'cache_used': cache_used,
            'logprobs': logprobs
            })