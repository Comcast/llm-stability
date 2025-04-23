import together
import os
import sys
from dotenv import load_dotenv
import helper_functions
from types import SimpleNamespace


load_dotenv()

from together import Together

MODEL = Together(api_key=os.environ["TOGETHER_KEY"])

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct-Turbo"

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
                      'rewrite_inst': str
                      }
    Returns:
       str: payload
       run_info: {'prompt':str, 
                  'model_run': 'model_template', 
                  'config': {'temperature':probability, ...}
                 }
    """
    cache_used = False
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
    logprobs_config = config.get('logprobs', False)
    response = MODEL.chat.completions.create(
                    messages=prompt,
                    model=MODEL_NAME,
                    temperature=config['temperature'],
                    seed=config['seed'],
                    top_p=config['top_p_k'],
                    logprobs=logprobs_config
                )
    #d = SimpleNamespace(a=1, b=2) Hack to keep interface to run_experiment.py
    if logprobs_config:
        logprobs = [SimpleNamespace(token=tk, logprob=lp) for tk, lp in \
                    zip(response.choices[0].logprobs.tokens,
                        response.choices[0].logprobs.token_logprobs)]
    else:
        logprobs = []
    return (response.choices[0].message.content,
            {
            'prompt':prompt, 
            'model_name': MODEL_NAME, 
            'temperature': config['temperature'],
            'seed': config['seed'],
            'top_p_k': config['top_p_k'],
            'rewrite_inst': config.get('rewrite_inst', None),
            'cache_used': cache_used,
            'logprobs': logprobs
            })