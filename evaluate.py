import argparse
import glob
import importlib
import itertools
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Callable
import pandas as pd
import datasets #not needed but forces check on modules being installed

sys.path.append(os.getcwd())

"""
Evaluation for code for experiments. 

Evaluates an experiment as configured by command line parameters. 

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

MODELS =['gpt-4o', 'llama-3-70b', 'finetuned-3.5', 'gpt-3.5-turbo',
        'mixtral-8x7b', 'llama-3-8b']
TASKS = ['high_school_european_history', 'college_mathematics',
         'geometric_shapes', 'navigate', 'professional_accounting', 
         'logical_deduction', 'ruin_names', 'public_relations']
SHOTS = ['0-shot', 'few_shot', '20-shot']

def load_runs(directory, old_format=False) -> pd.DataFrame:
    """
    Loads all .csv files in the specified directory recursively and expands likely
    serialized dictionaries from their respective columns.

    Args:
        directory (str): Directory containing .csv files.
        old_format (bool): If True, convert old format files to the new format.

    Returns:
        pd.DataFrame: DataFrame containing concatenated data from all .csv files.
    """
    files = glob.glob(os.path.join(directory, '**', '*.csv'), 
                        recursive=True)
    data_df = pd.DataFrame()
    
    for csv_file in files:
        csv_df = pd.read_csv(csv_file)
        print(f"Trying {csv_file}")
        if old_format:
            model = None
            for model_name in MODELS:
                if model_name in csv_file:
                    model = model_name
            
            assert model is not None
            csv_df['model'] = 'gpt-35-turbo' if model == 'gpt-3.5-turbo' \
                                else model
            model_config = {"temperature": 0.0, 
                            "seed": 12, 
                            "top_p_k": 0.0 if 'p0' in csv_file else 1.0}
            csv_df['model_config'] = json.dumps(model_config)
            task = None
            for task_name in TASKS:
                if task_name in csv_file:
                    task = task_name
            assert task is not None
            csv_df['task'] = task
            shot = None
            for shot_name in SHOTS:
                if shot_name in csv_file:
                    shot = shot_name if shot_name != 'few_shot' else 'few'
            assert shot is not None
            task_config = {"prompt_type": "v2", "shots": shot}
            csv_df['task_config'] = json.dumps(task_config)
            csv_df['prompt'] = '{}'
            csv_df['rubric'] = '{}'
            csv_df['rubric_id'] = csv_df.index
            csv_df['response'] = csv_df['raw_response']
            m = re.match(r'.*_(\d+)\.csv', csv_file)
            csv_df['run'] = int(m.group(1))
            csv_df['date'] = '2024-08-00_00-00-00' #Appromite date of runs
        for col in ['prompt', 'model_config', 'task_config', 'rubric']:
            csv_df[col] = csv_df[col].apply(lambda row: json.loads(row))
        csv_df['file'] = csv_file
        data_df = pd.concat([data_df, csv_df], ignore_index=True)
    return data_df


def get_experiment_configs(data_df: pd.DataFrame)-> list:
    """
    Returns all possible combinations of experiment configs, does
    not check that configs exist together in data_df. 
    Args:
        data_df (pd.DataFrame): Experiment runs 
    Returns:
        list: Tuples (model, model_config, task, model_config)
    """
    exp_configs = []
    for model in data_df['model'].unique():
        for model_config in data_df['model_config'].drop_duplicates():
            for task in data_df['task'].unique():
                for task_config in data_df['task_config'].drop_duplicates():
                    exp_configs.append((model, 
                                        model_config, 
                                        task, 
                                        task_config))
    return exp_configs

            
def check_hand_annotated_cache(row, answer_cache_df):
    """
    Check if a specific row exists in the answer cache DataFrame and return the parsed answer if found.
    Args:
        row (pd.Series): A pandas Series object representing a row of data with keys 'response', 'task', 'model', and 'rubric_id'.
        answer_cache_df (pd.DataFrame): A pandas DataFrame containing cached answers with columns 'response', 'task', 'model', 'id', and 'parsed_answer'.
    Returns:
        pd.Series or None: The 'parsed_answer' column from the matching row in answer_cache_df if a match is found, otherwise None is returned and the
        answer_cache_df is updated with a copy of the row that has 'parsed_answer' set to None. This will be serialized later for examination by an annotation process.
    """

    answer_df = \
        answer_cache_df[(answer_cache_df['response'] == row['response'])
                        & (answer_cache_df['task'] == row['task'])
                        & (answer_cache_df['model'] == row['model'])
                        & (answer_cache_df['rubric_id'] == row['rubric_id'])
                    ]
    if len(answer_df.index) == 1:
        return answer_df['parsed_answer'].iloc[0], answer_cache_df
    else:
        row_copy = \
            row[['response', 'task', 'model', 'rubric_id', 'rubric']].copy()
        row_copy['parsed_answer'] = None
        answer_cache_df = pd.concat([answer_cache_df, row_copy.to_frame().T], 
                                     ignore_index=True)
        
    return None, answer_cache_df

def remove_key_value_ret_dict(x:dict)->dict:
    """
        Removes 'rubric_counter' key/value from dict and returns dict
    """
    x.pop('rubric_counter', None)
    return x

def evaluate(data_df: pd.DataFrame, num_bootstrap_draws=10) -> (dict, pd.DataFrame, list):
    """
    Evaluates the experiment data by computing various metrics such as TARa, TARr, and correctness.

    This function processes the data grouped by model, model configuration, task, and task configuration.
    It calculates agreement counts, correctness, and bootstrap estimates for accuracy.

    NOTE: This code is meant to be clear rather than properly modularized. Hopefully the long format will make it clear how results are being accumulated and reported.

    Arguments:
        data_df (pd.DataFrame): DataFrame containing the experiment data. It is expected to have columns for model, model_config, task, task_config, and other relevant data.
        num_bootstrap_draws (int): Number of bootstrap samples to draw for estimating accuracy distributions.

    Returns:
        tuple: A tuple containing:
            - results (pd.DataFrame): DataFrame with evaluation results for each configuration.
            - data_df (pd.DataFrame): Updated DataFrame with additional columns for correctness and parsed answers.
            - errors (list): List of errors encountered during evaluation.
    """
    
    data_df['correct'] = False
    data_df['parsed_answer'] = None
    #remove rubric id used for fine tuned runs
    data_df['model_config'] = \
        data_df['model_config'].apply(lambda x: remove_key_value_ret_dict(x))
    configs = get_experiment_configs(data_df)
    results = pd.DataFrame()
    errors = []
    total_evals = 0
    task_x_rubric = set()
    if os.path.exists('answer_cache.csv'):
        answer_cache_df = pd.read_csv('answer_cache.csv')
    else:
        answer_cache_df = pd.DataFrame({'response':[], 'task': [], 'model': [],
                                        'parsed_answer':[], 'rubric':[], 'rubric_id':[]})
    for model, model_config, task, task_config in configs:
        try:
            task_module = importlib.import_module(f'tasks.{task}')
        except ModuleNotFoundError:
            print(f'Need to add {f"tasks.{task}"}, skipping eval')
            continue
        exp_df = data_df[(data_df['model'] == model)
                        & (data_df['model_config'] == model_config)
                        & (data_df['task'] == task)
                        & (data_df['task_config'] == task_config)]
        print(f"{model} {model_config} {task} {task_config}")
        if len(exp_df.index) == 0: #may have combos with no data
            continue
        total_agreement_count_raw = 0
        total_agreement_count_answer = 0
        num_runs = max(exp_df['run']) + 1
        correct = [0] * num_runs # need to track
        any_correct_count = 0
        any_wrong_count = 0
        correct_MACr = 0
        rubric_ids = exp_df['rubric_id'].unique()
        num_questions = len(rubric_ids)
        runs_accum = []
        correct_by_raw_count = [0] * num_runs
        incorrect_by_raw_count = [0] * num_runs
        for id in rubric_ids:
            seen_errors = defaultdict(int)
            task_x_rubric.add(f"{task}x{id}")
            question_df = exp_df[exp_df['rubric_id'] == id]
            #question_df = question_df.reset_index(drop=True)
            raw = set()
            answer = set()
            correct_raw_to_count = defaultdict(int)
            incorrect_raw_to_count = defaultdict(int)
            if not num_runs == len(question_df.index):
                print(f"{model}, {model_config}, {task}, {task_config}")
                error = f"runs not matching expected length, expected {num_runs}, got {len(question_df.index)} for {question_df['file'].to_list()}"
                raise IndexError(error)
            corrects_for_rubric = 0
            run_accum = [0] * num_runs
            for idx, row in question_df.iterrows(): # runs over question
                raw.add(task_module.raw_fn(row))
                total_evals += 1
                if pd.isna(row['response']):
                    errors.append(f"NaN response found {model} {task} {id}")
                    continue
                try:
                    parsed_answer = task_module.answer_fn(row, task_config)
                    if parsed_answer is None:
                        (parsed_answer, answer_cache_df) = \
                                check_hand_annotated_cache(row, answer_cache_df)
                    if parsed_answer is None:
                        error = f"No answer found {model} {task} {id}"
                        seen_errors[error] += 1
                        if seen_errors[error] == 1:
                            print(f"-----------------parse issue---{len(errors)}")
                            print(f"{error} {row['run']}")
                            print(f"Response: {row['response']}")
                            print(f"Rubric: {row['rubric']}")
                            errors.append(f"No answer found {model} {task} {id} {row['run']}")
                            answer.add(idx) # No answer will always fail TARa
                        print(f"No Answer Found: Repeat {seen_errors[error]} for rubric id {id}")
                        errors.append(f"No answer found {model} {task} {id} {row['run']}")
                        continue # cannot be correct so continue
                    else:
                        data_df.loc[idx,'parsed_answer'] = parsed_answer
                        answer.add(parsed_answer)
                except LookupError as e: #LookupError is 
                    answer.add(idx) # Blown UP is also a failure of TARa
                    data_df.loc[idx,'parsed_answer'] = "Blown UP"
                    error = f"Blown UP found {model} {task} {id} Answer: {e}"
                    seen_errors[error] += 1
                    if seen_errors[error] == 1:
                        print(f"------answer issue-----{len(errors)}---")
                        print(f"{error} {row['run']}")
                        print(f"Response: {row['response']}")
                        print(f"Rubric: {row['rubric']}")
                    errors.append(f"Blown UP {model} {task} {id} {row['run']}")
                    print(f"Blown UP: Repeat {seen_errors[error]} for rubric id {id}")
                    continue # can't be correct so continue
                if task_module.correct_fn(row, task_config):
                    correct[row['run']] += 1
                    run_accum[row['run']] = 1
                    data_df.loc[idx,'correct'] = True
                    corrects_for_rubric += 1
                    correct_raw_to_count[task_module.raw_fn(row)] += 1
                else:
                    incorrect_raw_to_count[task_module.raw_fn(row)] += 1
            if len(raw) == 1:
                total_agreement_count_raw += 1
            if len(answer) == 1:
                total_agreement_count_answer += 1
            if corrects_for_rubric != num_runs:
                any_wrong_count += 1
            if corrects_for_rubric > 0:
                any_correct_count += 1
            for count in correct_raw_to_count.values():
                correct_by_raw_count[count - 1] += 1
            for count in incorrect_raw_to_count.values():
                incorrect_by_raw_count[count - 1] += 1
            runs_accum.append(run_accum)
            # for N iterations, draw
        bootstrap_correct_counts = [0] * num_bootstrap_draws
        for d in range(num_bootstrap_draws):
            for run in runs_accum:
                bootstrap_correct_counts[d] += random.choice(run)
        bootstrap_pct = [c/num_questions for c in bootstrap_correct_counts]

        result_d = {'model': model, 
                    'model_config': model_config,
                    'task': task,
                    'task_config': task_config,
                    'TACr': total_agreement_count_raw,
                    'TARr': total_agreement_count_raw/num_questions,
                    'TACa': total_agreement_count_answer,
                    'TARa': total_agreement_count_answer/num_questions,
                    'correct_count_per_run': correct,
                    'correct_pct_per_run': [c/num_questions for c in correct],
                    'num_questions': num_questions,
                    'N': num_runs,
                    'best_possible_count': any_correct_count,
                    'best_possible_accuracy': any_correct_count/num_questions,
                    'worst_possible_count': num_questions-any_wrong_count,
                    'worst_possible_accuracy': (num_questions-any_wrong_count)/num_questions,
                    'spread': any_correct_count/num_questions - 
                              (num_questions-any_wrong_count)/num_questions,
                    'bootstrap_counts': sorted(bootstrap_correct_counts),
                    'bootstrap_pcts': sorted(bootstrap_pct),
                    'correct_by_raw_count': correct_by_raw_count,
                    'incorrect_by_raw_count': incorrect_by_raw_count,
                    'date': exp_df['date'].iloc[0]
        }
        result_s = pd.Series(result_d)
        results = \
            pd.concat([results, result_s.to_frame().T],  ignore_index=True)
        
        print((f"{len(seen_errors.keys()):,} rubrics had parsing problems for"
        + f" {len(errors):,} task x rubrics for {total_evals:,} total"
        + f" evaluations"))
    return results, data_df, errors

def format_to_pct(cell):
    """
    Converts a float or a list of floats to percentage string format.
    Args:
        cell (float or list): A float or a list of floats to be converted.
    Returns:
        str or list: A percentage string if input is a float, or a list of percentage strings if input is a list of floats. 
                     If the input is neither a float nor a list, it returns the input unchanged.
    """

    if isinstance(cell, float):
        return f"{cell:.1%}"
    elif isinstance(cell, list):
        return [f"{x:.1%}" if isinstance(x, float) else x for x in cell]
    else:
        return cell
    
if __name__ == "__main__":
    usage_message = ("python evaluate.py -d local_runs/"
                     + "\npython evaluate.py -h shows help message and more options")

    epilog_message = "Documentation for project is at: https://github.com/Comcast/llm-stability/blob/main/README.md"
    
    parser = argparse.ArgumentParser(usage=usage_message, epilog=epilog_message)

    parser.add_argument("-d", "--directory", required=True, 
                            help=("path to data dir to timestamp, e.g., " 
        + "experiments/low_temp/local_runs/10_tasks/2024-11-29_13-49-06/"))
    parser.add_argument("-eo", "--eval_orig", 
                        action="store_true",
                        required=False,
                        default=False,
                        help="Evaluate old run format from v2 paper")
    parser.add_argument("-npp", "--no_pretty_print_percentages", 
                        required=False,
                        default=False,
                        action="store_true",
                        help="Disables printing ratios as percentage with one digit of precision, keeps full precision and floats")
    command_args = parser.parse_args()
    data_df = load_runs(command_args.directory, command_args.eval_orig)
    
    (eval_df, dict, errors) = evaluate(data_df)
    if not command_args.no_pretty_print_percentages:
        eval_df = eval_df.map(format_to_pct)
    print(eval_df)
    eval_df.to_csv("stability_eval.csv")
    print("Open stability_eval.csv in your favorite spreadsheet for results")
