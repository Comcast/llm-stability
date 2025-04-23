import os
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(),'..', '..'))
import helper_functions
import run_experiment
from datetime import datetime, date, MINYEAR


MODELS = [
    #'mixtral-8x7b',
    #'gpt-35_OAI', 'gpt-4o_OAI', 'llama3-8b', 
    'deterministic-sim',
    'bimodal-sim'
    ]
MODEL_CONFIGS = [{'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
                  'logprobs':True, 
                  #'top_logprobs': 5
                  }]

YES_NO_TASKS = ['navigate']

A_K_TASKS = ['geometric_shapes']

A_D_TASKS = ['professional_accounting',  'college_mathematics',
         'logical_deduction', 'ruin_names',        
          'high_school_european_history',
         'public_relations'
        ]

TASKS = ['college_mathematics'] #


TASK_CONFIGS = [{'prompt_type': 'v2', 'shots': 0}]
                
experiments = helper_functions.experiment_setup(MODELS, MODEL_CONFIGS, 
                                                TASKS, TASK_CONFIGS)


for model, model_config, task, task_config in experiments:
    date = datetime.now()
    datetime_string = date.strftime("%Y-%m-%d_%H-%M-%S")
    #datetime_string = '0001-01-01_00-00-00'
    run_args = {'output_directory': 'runs/',
                'model': model,
                'model_config': model_config,
                'model_config_in_filename': model_config['temperature'],
                'task': task,
                'task_config': task_config,
                'task_config_in_filename': task_config['shots'],
                'num_runs': 5,
#                'limit_num_rubrics': 2
    }
    
    run_experiment.run(run_args, datetime_string)
    