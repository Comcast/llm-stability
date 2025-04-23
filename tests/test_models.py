import pytest
import os
import sys
sys.path.append(os.getcwd())
import tasks.professional_accounting
import tasks.navigate
import importlib
import json
import math
import numpy as np
"""
Testing models for basic processing and authentication. Usage:

`pytest tests/test_models.py`

To test single model, e.g., gpt-35-turbo:

`pytest tests/test_models.py::test_gpt_35_turbo`


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

def test_gpt_35_turbo():
    return #no longer have credentails
    model_name = 'gpt-35-turbo'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

    
def test_gpt_4o():
    return #no longer have credentails
    model_name = 'gpt-4o'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

def test_llama_8b():
    model_name = 'llama3-8b'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    #assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

def test_llama_70b():
    model_name = 'llama3-70b'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    #assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

def test_mixtral_8x7b():
    model_name = 'mixtral-8x7b'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    #assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

def test_gpt_4o_rewrite():
    return #no longer have credentails
    model_name = 'gpt-4o'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    rewrite_inst = ("Please rewrite the below prompt to maximize your"
                      + " understanding of the instruction for processing.")
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'rewrite_inst': rewrite_inst})
    assert len(run_info['prompt'][0]['content']) > 10
    assert run_info['prompt'] != test_prompt
    assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert run_info['rewrite_inst'] == rewrite_inst
    assert len(result) > 10
    assert not run_info['cache_used']

    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'rewrite_inst': rewrite_inst})
    assert run_info['cache_used']


def test_gpt_3_5_turbo_rewrite():
    return #no longer have credentails
    model_name = 'gpt-35-turbo'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    rewrite_inst = ("Please rewrite the below prompt to maximize your"
                      + " understanding of the instruction for processing.")
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'rewrite_inst': rewrite_inst})
    assert len(run_info['prompt'][0]['content']) > 10
    assert run_info['prompt'] != test_prompt
    assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert run_info['rewrite_inst'] == rewrite_inst
    assert len(result) > 10
    assert not run_info['cache_used']

    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'rewrite_inst': rewrite_inst})
    assert run_info['cache_used']

def test_gpt_3_5_turbo_prefix_suffix():
    return #no longer have credentails
    model_name = 'gpt-35-turbo'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    prefix = "x x x x x x "
    suffix = " y y y y y"
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'prefix': prefix,
                                            'suffix': suffix})
    assert run_info['prompt'][0]['content'] ==\
         f"{prefix}{test_prompt[0]['content']}{suffix}"

def test_gpt_4o_prefix_suffix():
    return #no longer have credentails
    model_name = 'gpt-4o'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    prefix = "x x x x x x "
    suffix = " y y y y y"
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'prefix': prefix,
                                            'suffix': suffix})
    assert run_info['prompt'][0]['content'] ==\
         f"{prefix}{test_prompt[0]['content']}{suffix}"


def test_gpt_4o_OAI():
    system_prompt = 'Please answer the following question with the answer field only.'
    schema = {
            "name": "yes_no",
            "schema":{ 
                "type": "object",
                "properties": {
                    "Answer": {
                    "type": "string",
                    "enum" : ["Yes", "No"]
                    }
                },
                "required": [
                    "answer"
                ]
            }
        }
    model_name = 'gpt-4o_OAI'
    llm = importlib.import_module(f'models.{model_name}')
    prompt = [{"role": "user", "content": "Is 835 even?"}]
    result, run_info = llm.run(prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'system_content': system_prompt,
                                            'answer_schema': schema})
    assert len(run_info['prompt'][0]['content']) > 0
    answer_d = json.loads(result)
    assert answer_d['Answer'] == 'No'

def test_gpt_35_OAI_fine_tuned():
    #pre trained fine tuned models
    model_config = {'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
                    'even_model': 'ft:gpt-3.5-turbo-0125:personal::BAJ2zmk1', 
                    'odd_model': 'ft:gpt-3.5-turbo-0125:personal::BAJA6sJq'}
    model_name = 'gpt-35_OAI'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    model_config['rubric_counter'] = 0
    result, run_info = llm.run(test_prompt, model_config)
    model_config['rubric_counter'] = 1
    result, run_info = llm.run(test_prompt, model_config)
    assert True

def test_gpt_35_OAI_logprob():
    #pre trained fine tuned models
    model_config = {'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
                    'logprobs':True}
    model_name = 'gpt-35_OAI'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0].token != ''
    assert run_info['logprobs'][0].logprob <= 0.0


def test_gpt_4o_OAI_logprob():
    #pre trained fine tuned models
    model_config = {'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
                    'logprobs':True}
    model_name = 'gpt-4o_OAI'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0].token != ''
    assert run_info['logprobs'][0].logprob <= 0.0


def test_llama3_8b_logprob():
    model_config = {'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
                    'logprobs':True}
    model_name = 'llama3-8b'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0].token != ''
    assert run_info['logprobs'][0].logprob <= 0.0


def test_mistral8x7b_logprob():
    model_config = {'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
                    'logprobs':True}
    model_name = 'mixtral-8x7b'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0].token != ''
    for tok in run_info['logprobs']:
        assert '▁' not in tok.token
    assert run_info['logprobs'][0].logprob <= 0.0

def test_deterministic_sim():
    # run python models/deterministic-sim.py to generate the expected results 
    # comprehensively.
    model_config = {'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
                    'logprobs':True, 'round': 0, 'rubric_counter': 0}
    model_name = 'deterministic-sim'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "IGNORED"}] #content is ignored
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0]['token'] != ''
    assert run_info['logprobs'][0]['logprob']<= 0.0

    model_config['round'] = 0 
    model_config['rubric_counter'] = 0
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0]['logprob'] == \
        math.log((model_config['rubric_counter'] + 1)/100)
    assert result == '(A) same'
    
    model_config['round'] = 0 
    model_config['rubric_counter'] = 20
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0]['logprob'] == \
        math.log((model_config['rubric_counter'] + 1)/100)
    assert result == f'(A) same'

    model_config['round'] = 0 
    model_config['rubric_counter'] = 99
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0]['logprob'] == \
        math.log((model_config['rubric_counter'] + 1)/100)
    assert result == f'(A) same'
    
    # test the 5/5 case
    model_config['round'] = 4
    model_config['rubric_counter'] = 0
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0]['logprob'] == \
        math.log((model_config['rubric_counter'] + 1)/100)
    assert result == '(A) different 4'

    model_config['round'] = 4
    model_config['rubric_counter'] = 99
    result, run_info = llm.run(test_prompt, model_config)
    assert run_info['logprobs'][0]['logprob'] == \
        math.log((model_config['rubric_counter'] + 1)/100)
    assert result == '(A) same'

def test_bimodal_sim():
    model_config = {'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
                    'logprobs':True, 'round': 0, 'rubric_counter': 0}
    model_name = 'bimodal-sim'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "IGNORED"}] #content is ignored
    result, run_info = llm.run(test_prompt, model_config)
    assert result == f'(A) different {model_config["round"]}'
    assert run_info['logprobs'][0]['logprob'] < np.log(.5)
    model_config['rubric_counter'] = 99
    result, run_info = llm.run(test_prompt, model_config)
    assert result == f'(A) same'
    assert run_info['logprobs'][0]['logprob'] > np.log(.5)



