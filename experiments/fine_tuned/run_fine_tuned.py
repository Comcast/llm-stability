import os
from openai import OpenAI
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
import helper_functions
import run_experiment
from datetime import datetime, date, MINYEAR
import random
import importlib
import time
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


MODELS = ['gpt-35_OAI']
MODEL_CONFIGS = [{'temperature': 0.0, 'seed': 12, 'top_p_k': 1.0}]
TASKS = ['navigate', 'logical_deduction']
#[, 'professional_accounting',        
#        'ruin_names', 'college_mathematics', 'high_school_european_history',
#        'public_relations', 'geometric_shapes'
#        ]

TASK_CONFIGS = [{'prompt_type': 'v2', 'shots': 'few', 'fine_tuned': True},
                # {'prompt_type': 'v2', 'shots': 0}
                ]


def write_jsonl(data_list: list, filename: str) -> None:
    """Pulled and modified from https://cookbook.openai.com/examples/how_to_finetune_chat_models
    """
    with open(os.path.join('fine_tuning_data', filename), "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

def upload_file(client, file_name: str, purpose: str) -> str:
    """Pulled and modified from https://cookbook.openai.com/examples/how_to_finetune_chat_models
    """
    with open(os.path.join('fine_tuning_data', file_name), "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose=purpose)
    return response.id

def write_upload_train(file_root, data, client, map_df)->str:
    training_start = 0
    training_end = 40
    validation_start = 40
    validation_end = 50
    validation_file =  f"{file_root}_validation.jsonl"
    write_jsonl(data[validation_start:validation_end], validation_file)
    validation_file_oai = \
        upload_file(client, validation_file, "fine-tune")
    training_file = f"{file_root}_train.jsonl"
    write_jsonl(data[training_start:training_end], training_file)
    training_file_oai = upload_file(client, training_file, "fine-tune")
    
    model = "gpt-3.5-turbo"
    print(f"Fine tuning {model} for {file_root}")
    method = {"type": "supervised", "supervised": {
                                    "hyperparameters": {"n_epochs": 1}
                                    }
             }
    
    if map_df is None:
        try: 
            map_df = pd.read_csv("model_map.csv")
        except FileNotFoundError:
            print("no model_map.csv, creating")
            map_df = pd.DataFrame({'file': []})
    file_df = map_df[map_df['file'] == file_root]
    if len(file_df.index) == 1:
        print(f"Already fine tuned model {file_root}")
        return (file_df['finetuned_model'].iloc[0], map_df)
    response = client.fine_tuning.jobs.create(
                                training_file=training_file_oai,
                                validation_file=validation_file_oai,
                                model=model,
                                method=method,
                                seed=12
                                )
    job_id = response.id
    print("Job ID:", response.id)
    print("Status:", response.status)
    done = False
    while (response.fine_tuned_model is None):
        print("-------------")
        time.sleep(10)
        events = client.fine_tuning.jobs.list_events(job_id).data
        events.reverse()
        for event in events:
            print(event.message)
        response = client.fine_tuning.jobs.retrieve(job_id)
    finetuned_model = client.fine_tuning.jobs.retrieve(job_id).fine_tuned_model
    run_data = {'file': file_root, 
                'finetuned_model': finetuned_model,
                'training_start': training_start, 
                'training_end': training_end, 
                'validation_start': validation_start, 
                'validation_end': validation_end, 
                'model': model, 
                'method': method,
                'validation_file_oai': validation_file_oai, 
                'training_file_oai': training_file_oai}
    df = pd.Series(run_data).to_frame().T
    map_df = pd.concat([map_df, df])
    map_df.to_csv("model_map.csv")
    print("appended and wrote model_map.csv")
    return (finetuned_model, map_df)


    
experiments = helper_functions.experiment_setup(MODELS, MODEL_CONFIGS, 
                                                TASKS, TASK_CONFIGS)
model_map = None
for model, model_config, task, task_config in experiments:
    date = datetime.now()
    #date = datetime(MINYEAR, 1, 1) #min value
    task_module = importlib.import_module(f'tasks.{task}')
    dev_data = task_module.get_test_data(task_config)
    print("Num examples:", len(dev_data))
    dataset_even = []
    dataset_odd = []
    for i, rubric in enumerate(dev_data): #build fine tuning data
        entry = {'messages':
                [{"role": "user", "content": rubric['input']},
                 {"role": "assistant", "content": rubric['target']}]}
        if i % 2 == 0:
            dataset_even.append(entry)
        else:
            dataset_odd.append(entry)
    
    client = OpenAI(api_key=os.environ['OPEN_AI_KEY'])
    task_to_use = task
    modle_config_in_file_name = model_config['temperature']
    if False: #if True then swap the fine tuned models
        if task == 'navigate':
            task_to_use = 'logical_deduction'
        if task == 'logical_deduction':
            task_to_use = 'navigate'
        modle_config_in_file_name = f"{model_config['temperature']}_finetuned_{task_to_use}"
        
    (even_model, model_map) = write_upload_train(f"{model}_{task_to_use}_even", 
                                     dataset_even, client, model_map)
    (odd_model, model_map) = write_upload_train(f"{model}_{task_to_use}_odd", 
                                     dataset_odd, client, model_map)
    model_config['even_model'] = even_model
    model_config['odd_model'] = odd_model

    datetime_string = date.strftime("%Y-%m-%d_%H-%M-%S")
    run_args = {'output_directory': 'runs',
                'model': model,
                'model_config': model_config,
                'model_config_in_filename': modle_config_in_file_name,
                'task': task,
                'task_config': task_config,
                'task_config_in_filename': task_config['shots'],
                'num_runs': 5,
                #'limit_num_rubrics': 2
    }
    val = run_experiment.run(run_args, datetime_string)
    if val == "Successfully run":
        break