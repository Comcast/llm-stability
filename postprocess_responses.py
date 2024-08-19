import re
import pandas as pd
from helper_functions import discard_text_after_answer, parse_file_name
import sys
import numpy as np
import os
import json
from collections import defaultdict

"""Takes a llm_output.csv file from an LLM run and parses out the answer and saves as parsed_llm_output.csv"""


def process_raw_response(row):
    response = row["raw_response"]
    if response is np.nan:
        return "none"
    if "the answer is no" in response.lower():  # for yes or no question e.g. navigate
        return "No"
    elif "the answer is yes" in response.lower():
        return "Yes"
    if "the only option" in response.lower():
        start_index = response.lower().rfind("the only option")
        response = response[start_index:]
    elif "the closest might be" in response.lower():
        start_index = response.lower().rfind("the closest might be")
        response = response[start_index:]
    elif "the closest match is" in response.lower():
        start_index = response.lower().rfind("the closest match is")
        response = response[start_index:]
    elif "none of the options" in response.lower():
        return "none"
    elif "none of the given options" in response.lower():
        return "none"
    elif "it is not any of the given options" in response.lower():
        return "none"
    elif "here is no correct answer from the provided options" in response.lower():
        return "none"
    elif "there might be a mistake in the question" in response.lower():
        return "none"
    elif "seems to be a mistake in the question" in response.lower():
        return "none"
    elif "there could be a mistake in the provided" in response.lower():
        return "none"
    elif "might be an error in the question" in response.lower():
        return "none"
    elif "might be a mistake in the provided" in response.lower():
        return "none"
    elif (
        "seems to be a mistake in the paragraph or the options provided"
        in response.lower()
    ):
        return "none"
    elif "does not exactly match any of the given options" in response.lower():
        return "none"
    elif "no single correct answer among the options" in response.lower():
        return "none"
    elif "answer cannot be determined based on the given options" in response.lower():
        return "none"
    elif "the given options cannot be correct" in response.lower():
        return "none"
    elif (
        "the question cannot be answered with the given information" in response.lower()
    ):
        return "none"
    elif "does not fit any of the given options" in response.lower():
        return "none"
    elif "the answer is not among the provided options" in response.lower():
        return "none"
    elif "answer is not available in the given options" in response.lower():
        return "none"
    elif "is not represented in the given options" in response.lower():
        return "none"
    elif "does not represent any of the given options" in response.lower():
        return "none"
    elif "does not match any of the given options" in response.lower():
        return "none"
    elif (
        "it is not possible to accurately select an answer from the given options"
        in response.lower()
    ):
        return "none"
    elif "the answer is not available in the given options" in response.lower():
        return "none"
    elif "answer is not among the options" in response.lower():
        return "none"
    elif (
        "there is no exact match for this path in the provided options"
        in response.lower()
    ):
        return "none"
    elif "the answer is not provided in the option" in response.lower():
        return "none"
    elif "it's impossible to determine what shape this path" in response.lower():
        return "none"
    elif "none of the options are humorous edits" in response.lower():
        return "none"
    elif "there is no humorous edit among the given options" in response.lower():
        return "none"
    elif ("**1." in response) and (
        "**2." in response
    ):  # the model tries to solve all examples including few shot
        start_index = response.find("**6.")
        response = response[start_index:]
    elif (
        ("1." in response)
        and ("2." in response)
        and ("3." in response)
        and ("4." in response)
        and ("5." in response)
        and ("6." in response)
    ):
        start_index = response.rfind("6.")
        if start_index == -1:
            start_index = response.rfind("5.")
        response = response[start_index:]
    elif (
        ("1." in response)
        and ("2." in response)
        and ("3." in response)
        and ("4." in response)
        and ("5." in response)
    ):
        start_index = response.rfind("5.")
        if start_index == -1:
            start_index = response.rfind("4.")
        response = response[start_index:]
    elif (
        ("1." in response)
        and ("2." in response)
        and ("3." in response)
        and ("4." in response)
    ):
        start_index = response.rfind("4.")
        if start_index == -1:
            start_index = response.rfind("3.")
        response = response[start_index:]
    elif "answer:" in response.lower():
        start_index = response.lower().rfind("answer:")
        response = response[start_index:]
    if "no, the response is not risky" in response.lower():
        return "no"
    elif "yes, the response is risky" in response.lower():
        return "yes"
    try:
        pred = re.findall("(?<=answer is indeed )(\S*)", response)[-1]
        pred = discard_text_after_answer(pred)
        return pred
    except:
        try:
            pred = re.findall("(?<= is )(\([ABCDEFGHIJKabcdefghijk]\))", response)[-1]
            pred = discard_text_after_answer(pred)
        except:
            try:
                pred = re.findall("(?<=answer is )(\S*)", response)[-1]
                pred = discard_text_after_answer(pred)
            except:
                try:
                    pred = re.findall(
                        "(?<= closest match is )(\([ABCDEFGHIJabcdefghij]\))", response
                    )[-1]
                    pred = discard_text_after_answer(pred)
                except:
                    try:
                        pred = re.findall(
                            "6. (\([ABCDEFGHIJKabcdefghijk]\))\S*", response
                        )[
                            -1
                        ]  # sometimes it is like 6. (C)
                        pred = discard_text_after_answer(pred)
                    except:
                        try:
                            pred = re.findall(
                                "A:The answer is (\([ABCDEFGHIJKabcdefghijk]\))\S*",
                                response,
                            )[-1]
                            pred = discard_text_after_answer(pred)
                        except:
                            try:
                                pred = re.findall(
                                    "(\([ABCDEFGHIJKabcdefghijk]\))\S*", response
                                )[-1]
                                pred = discard_text_after_answer(pred)
                            except:
                                try:
                                    pred = re.findall(
                                        "([ABCDEFGHIJKabcdefghijk]:)\S*", response
                                    )[-1]
                                    pred = pred.replace(":", "")
                                    pred = "(" + pred + ")"
                                    pred = discard_text_after_answer(pred)
                                except:
                                    try:
                                        pred = re.findall(
                                            "(?<=answer is )([YESyesNOno]+)", response
                                        )[-1]
                                        pred = discard_text_after_answer(pred)
                                    except:
                                        if (
                                            "so, we need to determine which one is the correct answer."
                                            in response.lower()
                                        ):
                                            return "none"
                                        elif "complex shape" in response.lower():
                                            return "none"


def post_process(file_or_dir, failed_parses):
    if os.path.isfile(file_or_dir):

        if file_or_dir.endswith(".csv"):
            print(f"Processing {file_or_dir}")
            raw_input_df = pd.read_csv(file_or_dir)
            # breakpoint()
            raw_input_df["new_extracted_pred"] = raw_input_df.apply(
                process_raw_response, axis=1
            )
            for i, row in raw_input_df.iterrows():
                if row["new_extracted_pred"] == "none":
                    failed_parses[os.path.basename(file_or_dir)].append(
                        row["raw_response"]
                    )
            parsed_dir = os.path.join(os.path.dirname(file_or_dir), "parsed")
            if not os.path.exists(parsed_dir):
                print(f"making dir {parsed_dir}")
                os.makedirs(parsed_dir)

            outfile = os.path.join(
                parsed_dir,
                os.path.basename(file_or_dir),
            )
            raw_input_df[["pred", "gt", "raw_response", "new_extracted_pred"]].to_csv(
                outfile
            )
            print(f"Wrote {outfile}")
    else:
        for sub_dir_or_file in os.listdir(file_or_dir):
            post_process(os.path.join(file_or_dir, sub_dir_or_file), failed_parses)


failed_parses = defaultdict(list)
post_process(sys.argv[1], failed_parses)
# for k, v in failed_parses.items():
#     print(f"{k} has {len(v)} failed parses below")
#     print(json.dumps(v, indent=4))
#     print(f"{k} has {len(v)} failed parses above")
