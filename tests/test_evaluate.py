import pytest
import os
import sys
sys.path.append(os.getcwd())
import evaluate
import pandas as pd
import json

"""
Tests evaluate function. 

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

def test_A_D():
    test_data = { #1 TARr perfect, TARa perfect
        'task':['professional_accounting'] * 2,
        'task_config': [{'prompt_type': 'v2'}] * 2,
        'run':list(range(2)),
        'model':['gpt-4o'] * 2,
        'model_config':[{}] *  2,
        'rubric_id':[1] * 2,
        'response':['(A)', '(A)'],
        'gt': ['(A)'] * 2,
        'date': ['some date'] * 2
        }
    df = pd.DataFrame(test_data)
    eval_df, data_df, errors_l = evaluate.evaluate(df)
    assert eval_df['TARr'].iloc[0] == 1.0
    assert eval_df['TARa'].iloc[0] == 1.0
    assert eval_df['correct_count_per_run'].iloc[0] == [1, 1]
    assert eval_df['correct_pct_per_run'].iloc[0] == [1.0, 1.0]
    assert data_df['correct'].to_list() ==  [True, True]
    assert data_df['parsed_answer'].to_list() ==  ['(A)', '(A)']


def test_v2_Yes_No():
    test_data = { #1 perfect output
        'task':['navigate'] * 3,
        'task_config': [{'prompt_type': 'v2'}] * 3,
        'run':list(range(3)),
        'model':['m1'] * 3,
        'model_config':[{}] *  3,
        'rubric_id':[1] * 3,
        'response':['Yes', 'Yes', 'Yes'],
        'gt': ['Yes'] * 3,
        'date': ['some date'] * 3
        }
    df = pd.DataFrame(test_data)
    eval_df, data_df, errors_l = evaluate.evaluate(df)
    assert eval_df['model'].iloc[0] == 'm1' #zero
    assert eval_df['TARr'].iloc[0] == 1.0
    assert eval_df['TARa'].iloc[0] == 1.0
    assert eval_df['correct_count_per_run'].iloc[0] == [1, 1, 1]
    assert eval_df['correct_pct_per_run'].iloc[0] == [1.0, 1.0, 1.0]
    assert eval_df['best_possible_count'].iloc[0] == 1
    assert eval_df['best_possible_accuracy'].iloc[0] == 1.0
    assert eval_df['worst_possible_count'].iloc[0] == 1
    assert eval_df['worst_possible_accuracy'].iloc[0] == 1.0
    assert eval_df['bootstrap_counts'].iloc[0] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert eval_df['bootstrap_pcts'].iloc[0] == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    

    test_data = { #2 TARr different, TARa perfect
        'task':['navigate'] * 2,
        'task_config': [{'prompt_type': 'v2'}] * 2,
        'run':list(range(2)),
        'model':['m1'] * 2,
        'model_config':[{}] *  2,
        'rubric_id':[1] * 2,
        'response':['The answer is yes ', 'Yes, Yes'],
        'gt': ['Yes'] * 2,
        'date': ['some date'] * 2
        }
    df = pd.DataFrame(test_data)
    eval_df, data_df, error_l = evaluate.evaluate(df)
    assert eval_df['TARr'].iloc[0] == 0.0
    assert eval_df['TARa'].iloc[0] == 1.0
    assert eval_df['correct_count_per_run'].iloc[0] == [1, 1]
    assert eval_df['correct_pct_per_run'].iloc[0] == [1.0, 1.0]
    assert eval_df['best_possible_count'].iloc[0] == 1
    assert eval_df['best_possible_accuracy'].iloc[0] == 1.0
    assert eval_df['worst_possible_count'].iloc[0] == 1
    assert eval_df['worst_possible_accuracy'].iloc[0] == 1.0
    assert eval_df['bootstrap_counts'].iloc[0] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert eval_df['bootstrap_pcts'].iloc[0] == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    test_data = { #3 TARa different
        'task':['navigate'] * 2,
        'task_config': [{'prompt_type': 'v2'}] * 2,
        'run':list(range(2)),
        'model':['m1'] * 2,
        'model_config':[{}] *  2,
        'rubric_id':[1] * 2,
        'response':['The answer is no ', 'Yes, Yes'],
        'gt': ['Yes'] * 2,
        'date': ['some date'] * 2
        }
    df = pd.DataFrame(test_data)
    eval_df, data_df, errors_l = evaluate.evaluate(df)
    assert eval_df['TARr'].iloc[0] == 0.0
    assert eval_df['TARa'].iloc[0] == 0.0
    assert eval_df['correct_count_per_run'].iloc[0] == [0, 1]
    assert eval_df['correct_pct_per_run'].iloc[0] == [0.0, 1.0]
    assert eval_df['best_possible_count'].iloc[0] == 1
    assert eval_df['best_possible_accuracy'].iloc[0] == 1.0
    assert eval_df['worst_possible_count'].iloc[0] == 0
    assert eval_df['worst_possible_accuracy'].iloc[0] == 0.0
    assert len(eval_df['bootstrap_counts'].iloc[0]) == 10
    assert len(eval_df['bootstrap_pcts'].iloc[0]) == 10

    test_data = { #4 TARa/TARr perfect, all wrong
        'task':['navigate'] * 2,
        'task_config': [{'prompt_type': 'v2'}] * 2,
        'run':list(range(2)),
        'model':['m1'] * 2,
        'model_config':[{}] *  2,
        'rubric_id':[1] * 2,
        'response':['No', 'No'],
        'gt': ['Yes'] * 2,
        'date': ['some date'] * 2
        }
    df = pd.DataFrame(test_data)
    eval_df, data_df, errors_l = evaluate.evaluate(df)
    assert eval_df['TARr'].iloc[0] == 1.0
    assert eval_df['TARa'].iloc[0] == 1.0
    assert eval_df['correct_count_per_run'].iloc[0] == [0, 0]
    assert eval_df['correct_pct_per_run'].iloc[0] == [0.0, 0.0] 
    assert eval_df['best_possible_count'].iloc[0] == 0
    assert eval_df['best_possible_accuracy'].iloc[0] == 0.0
    assert eval_df['worst_possible_count'].iloc[0] == 0
    assert eval_df['worst_possible_accuracy'].iloc[0] == 0.0
    assert len(eval_df['bootstrap_counts'].iloc[0]) == 10
    assert len(eval_df['bootstrap_pcts'].iloc[0]) == 10

def test_hand_annotated_cache():
    test_data_df = pd.DataFrame({ #1 TARr perfect, TARa perfect
        'task':['professional_accounting'] * 2,
        'task_config': [{'prompt_type': 'v2'}] * 2,
        'run':list(range(2)),
        'model':['gpt-4o'] * 2,
        'model_config':[{}] *  2,
        'rubric_id':[1] * 2,
        'rubric':['bala', 'blow'],
        'response':['blah blab', 'Something else'],
        'gt': ['(A)'] * 2,
        'date': ['some date'] * 2
        })

    answer_cache_df = pd.DataFrame({'response':[], 'task': [], 'model': [],
                                        'parsed_answer':[], 'rubric':[], 'rubric_id':[]})
    row = test_data_df.iloc[0]
    parsed_answer, answer_cache_df =\
         evaluate.check_hand_annotated_cache(row, answer_cache_df)
    assert len(answer_cache_df.index) == 1
    assert parsed_answer is None
    answer_cache_df.at[0, 'parsed_answer'] = 'the answer is X'
    parsed_answer, answer_cache_df =\
         evaluate.check_hand_annotated_cache(row, answer_cache_df)
    assert len(answer_cache_df.index) == 1
    assert parsed_answer == 'the answer is X'





