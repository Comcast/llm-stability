import math

"""
Model: Deterministic simulation of a model for testing purposes.
Authors: Breck Baldwin.

"""

MODEL = None
MODEL_NAME = "deterministic-sim"

def run(prompt: list, config: dict) -> (str):
    # Run `python models/deterministic-sim.py` to generate the expected results 
    # comprehensively.
    response_tokens = ['(', 'A', ')']
    quintile = .2 * config['round']
    
    prob = (config['rubric_counter'] + 1)/100
    logprob = math.log(prob)
    if prob > quintile:
        response_tokens.append(f' same')            
    else:
        response_tokens.append(f' different {config['round']}')        
        
    
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

if __name__ == "__main__":
    prompt = [{"role": "user", "content": "IGNORED"}] #content is ignored
    config = {'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
              'logprobs':True, 'round': 0, 'rubric_counter': 0}
    for round in range(5):
        for rubric_counter in range(100):
            config['round'] = round
            config['rubric_counter'] = rubric_counter
            result, run_info = run(prompt, config)
            print(f'{round} {rubric_counter} {result}')