import os
import sys
import helper_functions
import math
import numpy as np

"""

Model: Bimodal simulation of a model for testing purposes. Splits rubrics below 50 into low probabilities and TACr of 1/5, and above 50 into high probabilities in to TACr of 5/5.
Authors: Breck Baldwin.

"""

MODEL = None
MODEL_NAME = "deterministic-sim"

def run(prompt: list, config: dict) -> (str):
    
    response_tokens = ['(', 'A', ')']
    if config['rubric_counter'] > 50:
        response_tokens.append(f' same')
        prob = np.random.normal(1, 0.01, 1)[0]  # Centered at 0
    else:
        response_tokens.append(f' different {config['round']}')        
        prob = np.random.normal(0, 0.1, 1)[0]  # Centered at 1 
    logprob = np.log(np.clip(prob, 0.0001, 1))
    response = ''.join(response_tokens)
    response_logprobs =\
          [{'token': t, 'logprob': logprob} for t in response_tokens]

    return (response,
            {
            'prompt':prompt, 
            'model_name': MODEL_NAME, 
            'temperature': config['temperature'],
            'seed': config['seed'],
            'top_p_k': config['top_p_k'],
            'rewrite_inst': config.get('rewrite_inst', None),
            'cache_used': False,
            'logprobs': response_logprobs
            })
