from openai import AzureOpenAI, OpenAI
import os
import yaml
import pandas as pd
import random
import numpy as np
import torch
from datasets import Dataset

"""
Contains helper functions. 
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


def parse_file_name(file_name: str):
    """Splits file_name into model_name, task and experiment_seed
    arguments:
        file_name: str
    returns:
        model_name: str
        task: str
        experiment_seed: str
    """

    parsed_list = file_name.split("_")
    model_name = parsed_list[0]
    experiment_seed = parsed_list[-1].replace(".csv", "")
    task = "_".join(parsed_list[1:-1])
    return model_name, task, experiment_seed


def make_inference(model: AzureOpenAI | OpenAI, model_name: str, message: str):
    """Runs the LLM and returns response.
    arguments:
        model: AzuerOpenAI object
        model_name: str
        message: str data to be processed
    returns:
        str: raw response
    """
    if model_name == "databricks-mpt-30b-instruct":
        response = model.completions.create(message, model=model_name, temperature=0)
        raw_response = response.choices[0].text
        return raw_response
    else:
        response = model.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            model=model_name,
            temperature=0,
        )
        raw_response = response.choices[0].message.content
        return raw_response


def configure_model(model_name: str):
    """Returns object to run LLM.

    arguments:
        model_name: str
    returns:
        client: AzureOpenAI|OpenAI
    """
    # below will have to be specified
    OPENAI_API_TYPE = "azure"
    AZURE_ENDPOINT_GPT_3_5 = "https://"
    AZURE_ENDPOINT_GPT_4_0 = "https://"
    MIXTRAL_ETC_ENDPOINT = "https://"
    LLAMA_3_8b_ENDPOINT = "https://"

    if model_name == "gpt-3.5-turbo":
        deployment_id = "AppliedAI-gpt-35-turbo"
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT_GPT_3_5,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version="2023-07-01-preview",
            azure_deployment=deployment_id,
        )
    elif model_name == "gpt-4o":
        deployment_id = "AppliedAI-gpt-4o"
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT_GPT_4_0,
            api_key=os.getenv("OPENAI_GPT4_KEY"),
            api_version="2024-04-01-preview",
            azure_deployment=deployment_id,
        )
    elif model_name == "finetuned-3.5":
        deployment_id = "mmlu"
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT_GPT_3_5,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_deployment=deployment_id,
        )
    elif (
        (model_name == "llama-3-70b")
        or (model_name == "mixtral-8x7b-instruct")
        or (model_name == "mpt-30b")
    ):
        client = OpenAI(
            api_key=os.getenv("DATABRICKS_TOKEN"),
            base_url=MIXTRAL_ETC_ENDPOINT,
        )
    elif model_name == "llama-3-8b":
        client = OpenAI(
            api_key=os.getenv("LLAMA_3_8B_KEY"),
            base_url=LLAMA_3_8b_ENDPOINT,
        )
    return client


def get_fewshot_examples(task: str):
    """Pulls examples from indicated yaml file
    arguments:
        task: str name of yaml file to get few shots data from
    returns:
        data: list of all shot examples
    """
    with open(f"few_shot_examples/{task}.yaml") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        except Exception as e:
            print(e)
    return data


def convert_mmlu_data(data: Dataset):
    """Translates mmlu formatted data from numbers to letters and creates correct format for processing.
    arguments
        data: dict
    returns
        final_data: list of dict
    """
    final_data = []
    option_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    for example in data:
        options = "\n"
        for i, option in enumerate(example["choices"]):
            options = options + f"({option_mapping[i]}) {option}. "
        final_data.append(
            {
                "input": f"{example['question']}{options}",
                "target": f'({option_mapping[example["answer"]]})',
            }
        )
    return final_data


def set_seed(seed=123):
    """Sets various seeds to constant value.
    arguments:
        seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discard_text_after_answer(txt):
    """Parses out answer from response
    arguments
        txt: str LLM response text
    returns
        txt: str
    """
    if txt[-1] == ".":
        txt = txt[:-1]
    if txt[-1] == ",":
        txt = txt[:-1]

    if "(a)" in txt[:6].lower():
        return "(A)"
    elif "(b)" in txt[:6].lower():
        return "(B)"
    elif "(c)" in txt[:6].lower():
        return "(C)"
    elif "(d)" in txt[:6].lower():
        return "(D)"
    elif "(e)" in txt[:6].lower():
        return "(E)"
    elif "(f)" in txt[:6].lower():
        return "(F)"
    elif "(g)" in txt[:6].lower():
        return "(G)"
    elif "(h)" in txt[:6].lower():
        return "(H)"
    elif "(i)" in txt[:6].lower():
        return "(I)"
    elif "(j)" in txt[:6].lower():
        return "(J)"
    elif "(k)" in txt[:6].lower():
        return "(K)"
    else:
        return txt
