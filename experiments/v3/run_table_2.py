import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
import helper_functions
import run_experiment
from datetime import datetime, date, MINYEAR

MODELS = ['gpt-4o', 'gpt-35-turbo', 'llama3-8b', 'llama3-70b', 'mixtral-8x7b']
MODEL_CONFIGS = [{'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0}]
TASKS = ['navigate', 'logical_deduction', 'professional_accounting',        
        'ruin_names', 'college_mathematics', 'high_school_european_history',
        'public_relations', 'geometric_shapes'
        ]

TASK_CONFIGS = [{'prompt_type': 'v2', 'shots': 'few'},
                {'prompt_type': 'v2', 'shots': 0}]



experiments = helper_functions.experiment_setup(MODELS, MODEL_CONFIGS, 
                                                TASKS, TASK_CONFIGS)


for model, model_config, task, task_config in experiments:
    date = datetime.now()
    datetime_string = date.strftime("%Y-%m-%d_%H-%M-%S")
    run_args = {'output_directory': 'runs',
                'model': model,
                'model_config': model_config,
                'model_config_in_filename': model_config['temperature'],
                'task': task,
                'task_config': task_config,
                'task_config_in_filename': task_config['shots'],
                'num_runs': 10,
 #               'limit_num_rubrics': 2
    }
#    datetime_string = '0000'
    val = run_experiment.run(run_args, datetime_string)
    if val == "Successfully run":
        break