import os
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(),'..', '..'))
import helper_functions
import run_experiment
from datetime import datetime, date, MINYEAR


MODELS = ['gpt-4o_OAI']
MODEL_CONFIGS = [{'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0,
                  'system': 'Please answer the following question with the answer field only.'}]

YES_NO_TASKS = ['navigate']

A_K_TASKS = ['geometric_shapes']

A_D_TASKS = ['professional_accounting',  'college_mathematics',
         'logical_deduction', 'ruin_names',        
          'high_school_european_history',
         'public_relations'
        ]

TASKS = YES_NO_TASKS + A_K_TASKS + A_D_TASKS


TASK_CONFIGS = [{'prompt_type': 'v2', 'shots': 'few'}]
                # 'prefix': '\n\n', 'suffix': '\n\n'}]
                #{'prompt_type': 'v2', 'shots': 0}]

experiments = helper_functions.experiment_setup(MODELS, MODEL_CONFIGS, 
                                                TASKS, TASK_CONFIGS)


for model, model_config, task, task_config in experiments:
    date = datetime.now()
    datetime_string = date.strftime("%Y-%m-%d_%H-%M-%S")
    model_config['system_content'] = 'Please answer the following question with the answer field only.'
    if task in A_D_TASKS:
        enum = ["A", "B", "C", "D"]
    elif task in YES_NO_TASKS:
        enum = ["Yes", "No"]
    elif task in A_K_TASKS:
        enum = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    else:
        print(f"No schema for {task}, skipping")
        continue
    model_config['answer_schema'] = {
                                    "name": "answer_schema",
                                    "schema":{ 
                                        "type": "object",
                                        "properties": {
                                            "Answer": {
                                            "type": "string",
                                            "enum" : enum
                                            }
                                        },
                                        "required": [
                                            "answer"
                                        ]
                                    }
                                }

    run_args = {'output_directory': 'runs/',
                'model': model,
                'model_config': model_config,
                'model_config_in_filename': model_config['temperature'],
                'task': task,
                'task_config': task_config,
                'task_config_in_filename': task_config['shots'],
                'num_runs': 2,
                'limit_num_rubrics': 1
    }
    datetime_string = '0001-01-01_00-00-00'
    val = run_experiment.run(run_args, datetime_string)
    if val == "Successfully run":
        break