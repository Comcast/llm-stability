import pandas as pd
import os
import numpy as np
import sys

pd.options.display.float_format = "{:.6f}".format
from helper_functions import parse_file_name

"""Runs scoring. Example call:
`python calculate_statistics.py release_data/few_shot fewshot_scores`
Writes to 
`fewshot_scores/detail_accuracy.csv`: Parsed outputs for each model/task/run
`fewshot_scores/low_med_high.csv` Accuracy, low, median, high
`fewshot_scores/TARa.csv` TARa 
`fewshot_scores/TARr.csv` TARr

Note that mixtral-8x7b-instruct_bbh_navigate_2.csv and mixtral-8x7b-instruct_bbh_navigate_3.csv had the last row removed to match the 
counts of runs 0, 1 and 4. 

"""

"""
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


def check_element_wise_equality(lists):
    counter = 0
    for n_tuple in zip(*lists, strict=True):

        checker = True
        for i in range(len(n_tuple) - 1):
            for j in range(i, len(n_tuple)):
                if n_tuple[i] != n_tuple[j]:
                    checker = False
                    break
        if checker:
            counter += 1
    return counter


folder_path = sys.argv[1]
dest_folder = sys.argv[2]
if not os.path.exists(dest_folder):
    print(f"making dir {dest_folder}")
    os.makedirs(dest_folder)

all_tasks = set()
model_list = set()
for file_name in os.listdir(folder_path):
    if not file_name.endswith(".csv"):
        continue
    model_name, task, experiment_run = parse_file_name(file_name)
    all_tasks.add(task)
    model_list.add(model_name)

model_list = list(model_list)
results = {}
results_intersection = {}
results_raw_responses = {}
aggregated_results = {}

for model_name in model_list:
    results[model_name] = {}
    results_intersection[model_name] = {}
    results_raw_responses[model_name] = {}
    aggregated_results[model_name] = {}

last_run_name = ""
truth_count = -1
last_run_questions = []
for file_name in os.listdir(folder_path):
    if not file_name.endswith(".csv"):
        continue
    model_name, task, experiment_run = parse_file_name(file_name)
    run_name = f"{model_name} {task}"
    dump_file = pd.read_csv(f"{folder_path}/{file_name}")
    dump_file = dump_file.fillna("")  #
    targets = dump_file["gt"]
    if run_name != last_run_name:  # reset data checks with change of run_name
        last_run_name = run_name
        truth_count = len(targets)
        last_run_questions = []
    if len(targets) != truth_count:
        raise RuntimeError(
            f"Truth counts has changed, had {truth_count}, got {len(targets)} for {file_name}"
        )
    # Two passes of prediction are available, we look at both
    # "new_extracted_pred" comes from running `postprocess_responses.py`
    preds_1 = dump_file["pred"].apply(lambda x: x.lower().strip().replace(".", ""))
    preds_2 = dump_file.get(
        "new_extracted_pred", pd.Series(data=[""] * len(targets))
    ).apply(lambda x: x.lower().strip().replace(".", ""))
    truths = dump_file["gt"].apply(lambda x: x.lower().strip())
    # preds = (
    #     dump_file["new_extracted_pred"]
    #     if "new_extracted_pred" in dump_file.columns
    #     else dump_file["pred"]
    # )
    # not great, easy for bugs to get through.
    correct = 0
    total = 0
    preds_1_counter = 0
    used_preds = []
    # if file_name == "llama-3-70b_bbh_navigate_0.csv":
    #    breakpoint()
    for i, truth in enumerate(truths):
        if "question" in dump_file:  # test that questions match if there
            question = dump_file["question"].iloc[i]
            if i == len(last_run_questions):
                last_run_questions.append(question)
            elif question != last_run_questions[i]:
                raise RuntimeError(
                    f"Question does not match other runs {i} in {file_name}"
                )
        if truth == preds_2.iloc[i]:
            correct += 1
            used_preds.append(preds_1.iloc[i])
        elif truth == preds_1.iloc[i]:
            correct += 1
            preds_1_counter += 1
            used_preds.append(preds_2.iloc[i])
        else:
            used_preds.append(
                f"Wrong pred1: {preds_1.iloc[i]} .... pred2: {preds_2.iloc[i]}"
            )
        total += 1
    acc = correct / total
    if preds_1_counter > 0:
        print(f"** Preds 1 {preds_1_counter}/{total} {file_name}")

    if task in results[model_name]:  # really bad, need defaultdict here
        results[model_name][task].append(100 * acc)
        results_intersection[model_name][task].append(used_preds)
        results_raw_responses[model_name][task].append(
            [response.lower().strip() for response in dump_file["raw_response"]]
        )
    else:
        results[model_name][task] = [100 * acc]
        results_intersection[model_name][task] = [used_preds]
        results_raw_responses[model_name][task] = [
            [response.lower().strip() for response in dump_file["raw_response"]]
        ]

# ACCURACY FOR EACH FILE DUMP
tasks = []
runs = []
models = []
accs = []

for model_name in results:
    for task in results[model_name]:
        tasks.extend([task] * len(results[model_name][task]))
        runs.extend(list(range(len(results[model_name][task]))))
        models.extend([model_name] * len(results[model_name][task]))
        accs.extend(results[model_name][task])


pd.DataFrame({"model": models, "run": runs, "task": tasks, "accuracy": accs}).to_csv(
    os.path.join(dest_folder, "detail_accuracy.csv")
)

# Calculate mean std etc.
for model_name in results:
    for task in results[model_name]:
        median = np.median(results[model_name][task])
        low = np.min(results[model_name][task])
        high = np.max(results[model_name][task])
        aggregated_results[model_name][task] = (low, median, high)

# NOW CREATE A DATAFRAME
tasks = []
mean_results = {}
std_results = {}
var_results = {}
intersection_percentages = {}
raw_intersection_percentages = {}

for model_name in model_list:
    mean_results[model_name] = []
    std_results[model_name] = []
    var_results[model_name] = []
    intersection_percentages[f"{model_name}_TARa"] = []
    raw_intersection_percentages[f"{model_name}_TARr"] = []


for task in all_tasks:
    tasks.append(task)
    for model in model_list:  # ["gpt-4o", "gpt-3.5-turbo"]: #model_list:
        mean_results[model].append(aggregated_results[model][task][0])
        std_results[model].append(aggregated_results[model][task][1])
        var_results[model].append(aggregated_results[model][task][2])

df_dict = {"task": tasks}
for model in model_list:  # ["gpt-4o", "gpt-3.5-turbo"]: #model_list:
    df_dict[f"{model}_low"] = mean_results[model]
    df_dict[f"{model}_median"] = std_results[model]
    df_dict[f"{model}_high"] = var_results[model]


results_df = pd.DataFrame(df_dict)
results_df.to_csv(os.path.join(dest_folder, "low_med_high.csv"))  # _0shotnoinst

# Calculate intersection results
tasks = []

for task in all_tasks:
    tasks.append(task)
    for model in model_list:  # ["gpt-4o", "gpt-3.5-turbo"]: #model_list:
        lists = results_intersection[model][task]
        try:
            total_intersection_count = check_element_wise_equality(lists)
        except:
            breakpoint()
        intersection_percentages[f"{model}_TARa"].append(
            100 * total_intersection_count / len(lists[0])
        )

answ_TAR_df = pd.DataFrame({"task": tasks, **intersection_percentages})
answ_TAR_df = answ_TAR_df.reindex(sorted(answ_TAR_df.columns), axis=1)
answ_TAR_df = answ_TAR_df.to_csv(os.path.join(dest_folder, "TARa.csv"))


# Calculate exact match intersection
tasks = []

for task in all_tasks:
    tasks.append(task)
    for model in model_list:  # ["gpt-4o", "gpt-3.5-turbo"]: #model_list:
        lists = results_raw_responses[model][task]
        total_intersection_count = check_element_wise_equality(lists)
        raw_intersection_percentages[f"{model}_TARr"].append(
            100 * total_intersection_count / len(lists[0])
        )

raw_TAR_df = pd.DataFrame({"task": tasks, **raw_intersection_percentages})
raw_TAR_df = raw_TAR_df.reindex(sorted(raw_TAR_df.columns), axis=1)
raw_TAR_df.to_csv(os.path.join(dest_folder, "TARr.csv"))
