from openai import OpenAI
import os
import sys
import helper_functions
from dotenv import load_dotenv

import json
from textwrap import dedent

load_dotenv()

"""
Model: GPT-4o hosted by OpenAI, see implementation for details on version.
Want to try the `strict=True` setting as presented in https://cookbook.openai.com/examples/structured_outputs_intro.

Authors: Breck Baldwin.
How to setup model to run: Model requires Azure Open AI authentication and endpoints as shown in code. If you need to setup an Azure Open AI account go to: https://azure.microsoft.com/en-us/products/ai-services/openai-service/. 
Tests: Run `pytest tests/test_models.py::test_gpt_4o` to verify that model functions. 


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
MODEL_NAME = "gpt-4o"

seen_before = {} #cache previous results for run to avoid variation

def run(prompt: list, config: dict) -> (str):
    """
    Runs a prompt and returns the content portion of the response as well as
    the actual run configuration. 
    Arguments:
       prompt: list of dicts, example: [{"role": "user", "content": question}]
       config: dict, {'temperature': float probability,
                      'seed': int,
                      'top_p_k': float probability,
                      'rewrite_inst': str| None,
                      'prefix': str| None,
                      'suffix': str| None
                      }
    Returns:
       str: payload
       run_info: {'prompt':str, 
                  'model_run': 'model_template', 
                  'config': {'temperature':probability, ...}
                 }
    """
    global seen_before
    cache_used = False
    temperature = config['temperature']
    seed = config['seed']
    top_p = config['top_p_k']
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
    if config.get('answer_schema', None) is not None:
        prompt.insert(0, {"role": 'system',
                    "content": config['system_content']})
        response = \
            MODEL.chat.completions.create(model=MODEL_NAME,
                                        messages=prompt,
                                        response_format={
                                            "type": "json_schema",
                                            "json_schema": config['answer_schema']
                                        })
    else:
        response = MODEL.chat.completions.create(
                    messages=prompt,
                    model=MODEL_NAME,
                    temperature=temperature,
                    seed=seed,
                    top_p=top_p,
                    logprobs=logprobs_config,
                    top_logprobs=config.get('top_logprobs', 0)
                )
    if logprobs_config:
        logprobs = response.choices[0].logprobs.content
    else:
        logprobs = []
    return (response.choices[0].message.content,
            {
            'prompt':prompt, 
            'model_name': MODEL_NAME, 
            'temperature': temperature,
            'seed': seed,
            'top_p_k': top_p,
            'rewrite_inst': config.get('rewrite_inst', None),
            'cache_used': cache_used,
            'logprobs': logprobs
            })
