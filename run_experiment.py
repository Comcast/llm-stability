from datasets import load_dataset
import tqdm
import importlib
import os
import sys
import argparse
import shutil
import glob
import json
import pandas as pd
from datetime import datetime, date, MINYEAR
import openai

#find models/tasks dir

"""
Runs a given experiment as configured by command line parameters. 

@authors Breck Baldwin

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


def run_model(llm, 
              out_dir: str, 
              outfile_root: str, 
              num_runs: int,
              model_config: dict, 
              test_rubrics: list,
              model_name: str,
              task_name: str,
              task_config: dict,
              date: str,
              context=None
             )-> str:
    """ 
    Runs model as specified. Designed to allow for re-running existing 
    experiment with no dependency on `tasks` data source since they can 
    are often external data sources like Hugging Face `datasets`. 
    Arguments:
        llm: Any module that has a `run` function
        out_dir: str, output directory
        outfile_root: str, usually MODEL_NAME-TASK_NAME 
        num_runs: int, how many runs to execute
        model_config: dict, configuration info to llm 
        task_config: dict, configuration info to task
        date: str, second resolution when run was invoked
    Side Effects:
        Writes data to disk in the form: `out_dir/<date_human_readable>/outfile_root-RUN_NUM.csv`
    """
    for i in range(num_runs):
        if context is not None:
            context.markdown(f"Running run {i}")
        llm_responses = []
        ground_truths = []
        questions = []
        modified_questions = []
        prompts = []
        model_configs = []
        rubrics = [] # entire test payload as serialized json
        models = []
        tasks = []
        task_configs = []
        runs = []
        rubric_ids = []
        dates = []
        logprobs = []
        print(f"Running {i}")
        rubric_counter = 0
        for rubric in tqdm.tqdm(test_rubrics):
            if context is not None:
                if rubric_counter % 50 == 0 or rubric_counter == 10:
                    context.markdown(f"Running rubric {rubric_counter}")
            prompt = [{"role": "user", "content": rubric['input']}]
            model_config['rubric_counter'] = rubric_counter
            model_config['round'] = i
            try:
                response, run_config = llm.run(prompt, model_config)
                for config in ['temperature', 'top_p_k', 'seed']:
                    if config in model_config:
                        assert run_config[config] == model_config[config]
            except openai.BadRequestError as e:
                response = "LLM Error"
                print(f"*****PROCESSING ERROR {e} for {rubric}")
            prompts.append(json.dumps(run_config['prompt']))
            rubrics.append(json.dumps(rubric))
            questions.append(rubric['input'])
            modified_questions.append(run_config['prompt'][0]['content'])
            ground_truths.append(rubric['target'])
            del model_config['rubric_counter']
            del model_config['round']
            model_configs.append(json.dumps(model_config))
            llm_responses.append(response)
            models.append(model_name)
            tasks.append(task_name) 
            task_configs.append(json.dumps(task_config))
            runs.append(i)
            rubric_ids.append(rubric_counter)
            dates.append(date)
            logprobs.append(json.dumps([{'token':e['token'], 'logprob':e['logprob']} \
                for e in run_config.get('logprobs', [])], indent=4))
            rubric_counter += 1
        df = pd.DataFrame({
                            'model': models,
                            'model_config':model_configs,
                            'task': tasks,
                            'task_config': task_configs,
                            'rubric': rubrics,
                            'rubric_id': rubric_ids,
                            'question': questions, 
                            'modified_questions': modified_questions,
                            'gt': ground_truths,
                            'prompt':prompts, 
                            'run': runs,
                            'response': llm_responses, 
                            'logprobs': logprobs,
                            'date': dates})
        assert len(df.index) == rubric_counter
        run_file = os.path.join(out_dir, f"{outfile_root}-{i}.csv")
        df.to_csv(run_file)
        print(f"*** Wrote {run_file}")
    return "Successfully run"

def run(run_args: dict, date_str: str) -> str:
    """
    Runs one model against one task n times and writes to specified output 
    to a path in the format `<outpath>/<model_name>-<task name>-<run num>.csv`

    Arguments:
        run_args: dict, configuration for the run. Look at argparse for possible and required values. No checking done here. 
    """
    ran_experiments =\
        glob.glob(os.path.join(run_args['output_directory'],'**'),
                  recursive=True)
    model_module_name = run_args['model'].replace('models/','').\
        replace('models\\','').replace('.py', '')
    task_module_name = run_args['task'].replace('tasks/','').\
        replace('tasks\\','').replace('.py', '')
    outfile_root = (f"{model_module_name}"
                    + f"-{run_args.get('model_config_in_filename', '')}"
                    + f"-{task_module_name}"
                    + f"-{run_args.get('task_config_in_filename', '')}")
    existing_runs = [f for f in ran_experiments if outfile_root in f]
    
    if len(existing_runs) != 0 and date_str != '0001-01-01_00-00-00':
        print(f"Already run {outfile_root},skipping")
        return "already run"
    print(f"No existing file, running {outfile_root}")

    llm = importlib.import_module(f'models.{model_module_name}')
    print("Model loaded")
    
    task_module = importlib.import_module(f'tasks.{task_module_name}')
    test_rubrics = task_module.get_test_data(run_args['task_config'])
    if 'limit_num_rubrics' in run_args:
        print(f"Limiting test to first {run_args['limit_num_rubrics']} rubrics")
        test_rubrics = test_rubrics[:run_args['limit_num_rubrics']]
    print("Data loaded")
    out_date_dir = os.path.join(run_args['output_directory'],
                            f"{outfile_root}_{date_str}")
    print(f"Making output directories if necessary {out_date_dir}")
    os.makedirs(out_date_dir, exist_ok=True)
    #outfile_root = f"{model_module_name}-{task_module_name}"
    #model_config = run_args['model_config']
    
    print("File system ready for run")
    run_model(llm=llm,
              out_dir=out_date_dir, 
              outfile_root=outfile_root, 
              num_runs=run_args['num_runs'],
              model_config=run_args['model_config'], 
              test_rubrics=test_rubrics,
              model_name=model_module_name,
              task_name=task_module_name,
              task_config=run_args['task_config'],
              date=date_str)
    return out_date_dir
    

if __name__ == "__main__":
    
    usage_message = ("python run_experiment.py -m gpt-4o -mc '{\"temperature\":0.0, \"seed\": 12, \"top_p_k\": 0.0}' -t navigate -tc '{\"prompt_type\": \"v2\", \"shots\": 0}' -n 2 -l 3 -et"
    +  "\npython run_experiment.py -h shows help message and more options")

    epilog_message = "Documentation for project is at: https://github.com/Comcast/llm-stability/blob/main/README.md"

    sys.path.append(os.path.join(os.getcwd(), 'models'))
    sys.path.append(os.path.join(os.getcwd(), 'tasks'))    
    if not os.path.exists("local_runs"):
        print((f"Stopping, expecting local_runs/ in working directory," 
              + f" have {os.getcwd()}, run as `python run_experiment.py`?"))
    parser = argparse.ArgumentParser(usage=usage_message, epilog=epilog_message)
    parser.add_argument("-m", "--model", required=True, 
                            help="Name of module in models/")
    parser.add_argument("-mc", "--model_config", type=str,
                        required=True,
                            help="Configuration for model")
    parser.add_argument("-t", "--task", required=True, 
                        help="Name of task module in tasks/")
    parser.add_argument("-tc", "--task_config", required=True,
                        help="Configuration for task")
    parser.add_argument("-n", "--num_runs", type=int, required=True, 
                        help="Number of runs to execute")
    parser.add_argument("-d", "--output_directory", required=False,
                default="local_runs",
                help="Where to write output files, will create all directories")
    parser.add_argument("-l", "--limit_num_rubrics", type=int)
    parser.add_argument("-et", "--use_earliest_time_stamp", action="store_true",
                        help="Creates time stamp: 0001-01-01_00-00-00. Will overwrite previous run.")
    command_line_args = parser.parse_args()
    date = datetime.now()
    if command_line_args.use_earliest_time_stamp:
        date = datetime(MINYEAR, 1, 1) #min value
    datetime_string = date.strftime("%Y-%m-%d_%H-%M-%S")
    command_line_args_d = vars(command_line_args)
    command_line_args_d['model_config'] =\
         json.loads(command_line_args_d['model_config'])
    command_line_args_d['model_config_in_filename'] =\
        command_line_args_d['model_config']['temperature']
    command_line_args_d['task_config'] =\
         json.loads(command_line_args_d['task_config'])
    command_line_args_d['task_config_in_filename'] =\
        command_line_args_d['task_config']['shots']
    out_dir = run(command_line_args_d, date_str=datetime_string)
    print(f"Run successful, run `python evaluate.py -d {out_dir}")
