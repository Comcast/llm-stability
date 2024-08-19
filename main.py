import argparse
from datasets import load_dataset
from tqdm import tqdm
import re
import pandas as pd
from helper_functions import (
    get_fewshot_examples,
    convert_mmlu_data,
    configure_model,
    set_seed,
    discard_text_after_answer,
    make_inference,
)

"""Top level main function to run LLMs."""

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

parser = argparse.ArgumentParser(description="")
parser.add_argument("--task", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--num_fewshot", type=int)
parser.add_argument("--experiment_run", type=int)

args = parser.parse_args()
set_seed(123)

if "bbh" in args.task:
    subset_name = args.task.split("_")
    subset_name = "_".join(subset_name[1:])
    test_data = load_dataset("lukaemon/bbh", name=subset_name, split="test")
elif "mmlu" in args.task:
    subset_name = args.task.split("_")
    subset_name = "_".join(subset_name[1:])
    test_data = load_dataset("cais/mmlu", name=subset_name, split="test")
    test_data = convert_mmlu_data(test_data)
else:
    raise Exception(f"{args.task} is not supported")

if "llama-3-70b" == args.model:
    model_name = "databricks-meta-llama-3-70b-instruct"
elif "mixtral-8x7b-instruct" == args.model:
    model_name = "databricks-mixtral-8x7b-instruct"
elif "finetuned-3.5" == args.model:
    model_name = "mmlu"
else:
    model_name = args.model

model_client = configure_model(args.model)
few_shot_data = get_fewshot_examples(subset_name)
tokenizer = None

raw_responses = []
predictions = []
ground_truths = []
need_check = False

raw_inputs = []

for example in tqdm(test_data):
    few_shot_text = ""
    for ex in few_shot_data["samples"][: args.num_fewshot]:
        few_shot_text = few_shot_text + ex["input"] + f"A:{ex['target']}\n"
    message = f"{few_shot_text} {example['input']} A:"
    raw_response = make_inference(model_client, model_name, message)
    raw_inputs.append(message)
    ground_truths.append(example["target"])
    raw_responses.append(raw_response)
    try:
        pred = re.findall("(?<=answer is )(\S*)", raw_response)[0]
        pred = discard_text_after_answer(pred)
    except:
        try:
            pred = re.findall("(?<= is )(\([ABCDabcd]\))(?=.)", raw_response)[0]
            pred = discard_text_after_answer(pred)
        except:
            need_check = True
            print("CHECK the Raw responses!!!")
            pred = raw_response
    predictions.append(pred)


pd.DataFrame(
    {
        "pred": predictions,
        "gt": ground_truths,
        "raw_response": raw_responses,
        "raw_input": raw_inputs,
    }
).to_csv(f"{args.model}_{args.task}_{args.experiment_seed}.csv")
