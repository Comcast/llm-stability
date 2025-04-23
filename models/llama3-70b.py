import together
import os
import sys
from dotenv import load_dotenv
import helper_functions

load_dotenv()

from together import Together

MODEL = Together(api_key=os.environ["TOGETHER_KEY"])

MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct-Turbo"

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
    response = MODEL.chat.completions.create(
                    messages=prompt,
                    model=MODEL_NAME,
                    temperature=config['temperature'],
                    seed=config['seed'],
                    top_p=config['top_p_k']
                )
    return (response.choices[0].message.content,
            {
            'prompt':prompt, 
            'model_name': MODEL_NAME, 
            'temperature': config['temperature'],
            'seed': config['seed'],
            'top_p_k': config['top_p_k'],
            'rewrite_inst': config.get('rewrite_inst', None),
            'cache_used': cache_used,
            'logprobs': []
            })