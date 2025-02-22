import torch
import transformers

MODEL_NAME ="meta-llama/Meta-Llama-3-8B-Instruct"

MODEL = transformers.pipeline(
    "text-generation", model=MODEL_NAME, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

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
    if config.get('prefix', None) is not None:
        prompt = config['prefix'] + prompt[0]['content']
    if config.get('suffix', None) is not None:
        prompt = prompt[0]['content'] + config['suffix']
    
    torch.manual_seed(config['seed'])
    response = MODEL(prompt, 
                     temperature=config['temperature'], 
                     top_p=config['top_p_k'])
    
    return (response[0]["generated_text"][1]["content"],
            {
            'prompt':prompt, 
            'model_name': MODEL_NAME, 
            'temperature': config['temperature'],
            'seed': config['seed'],
            'top_p_k': config['top_p_k'],
            'rewrite_inst': config.get('rewrite_inst', None),
            'cache_used': False
            })