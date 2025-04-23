import pytest
import os
import sys
sys.path.append(os.getcwd())
import run_experiment
import shutil
from datetime import datetime, date, MINYEAR

def test_end_to_end_run(): 
    out_dir = os.path.join('local_runs', 'tmp_test')
    shutil.rmtree(out_dir, ignore_errors=True)
    model = 'llama3-8b'
    task = 'professional_accounting'
    args = {'model': model, #module name version
            'model_config': {"temperature":0.0, "seed": 12, "top_p_k": 0.0}, 
            'task': f'tasks/{task}.py', #file path version
            'task_config': {"prompt_type": "v2", "shots":"few"}, 
            'num_runs': 1, 
            'output_directory': out_dir, 
            'limit_num_rubrics': 1,
            'use_earliest_time_stamp': True}
    #set time to earliest possible time--likely dev mode when working on code
    datetime_string = datetime(MINYEAR, 1, 1).strftime("%Y-%m-%d_%H-%M-%S")
    run_experiment.run(args, datetime_string)
    # output created    
    outfile = os.path.join(out_dir, f'{model}--{task}-_{datetime_string}')
    assert os.path.exists(outfile)
    shutil.rmtree(os.path.join(out_dir))

def test_backslash_removal():
    """
    Test for removing backslash from args on windows machines
    """ 
    out_dir = os.path.join('local_runs', 'tmp_test')
    shutil.rmtree(out_dir, ignore_errors=True)
    model = 'llama3-8b'
    task = 'professional_accounting'
    
    args = {'model': model, #module name version
            'model_config': {"temperature":0.0, "seed": 12, "top_p_k": 0.0}, 
            'task': f'tasks/{task}.py', #file path version
            'task_config': {"prompt_type": "v2", "shots":"few"}, 
            'num_runs': 1, 
            'output_directory': out_dir, 
            'limit_num_rubrics': 1,
            'use_earliest_time_stamp': True}
    datetime_string = datetime(MINYEAR, 1, 1).strftime("%Y-%m-%d_%H-%M-%S")
    run_experiment.run(args, datetime_string) 
    # output created
    outfile = os.path.join(out_dir, f'{model}--{task}-_{datetime_string}')
    assert os.path.exists(outfile)
    shutil.rmtree(os.path.join(out_dir))

    