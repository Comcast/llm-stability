# LLM Stability Project:

Public repo: https://github.com/Comcast/llm-stability

v3.0

## 1. Supporting software and data

This repo contains source code and data used to conduct the experiments reported in the papers below.

- v1.0 is the original ArXive paper v1.0 Submitted July 26, 2024 ArXiv released August 6, 2024: https://arxiv.org/abs/2408.04667v1. There is no support for this paper in the repo.
- v2.0 (v2.0 https://arxiv.org/abs/2408.04667v2) is a superset of the v1.0 work and where effort has been made to release software and data in a reproducible way. The release was a retrofit onto an existing project. The repo has a release 1.0 that reflects this version of the work--apologies for the confusing naming. 
- v3.0 extends the v2.0 experiments and analysis and has a completely new code base that focuses on reproducibility and clarity behind the experiments in the code base.  

## 2. Overview 

The repo contains raw data from experiment runs, software to run experiments and evaluate the experiments. In more detail:

- Data: The tar files `experiments/v2/runs.tgz` contain data used to generate Table 2 in `LLM_Stability_paper_v2.0.pdf` and was created with the v1.0 release of the repo infrastructure. The data `experiments/v3/runs.tgz` recreates the v2 data with the `run_experiment.py` but with 10 runs.
- Running experiments: The script `run_experiment.py` loads a model, task and produces time stamped data in the `local_runs` directory. Details below. 
- Evaluation: The script `evaluate.py` parses the output of the experiment runs and calculates model performance which results in an `evaluation_output.csv` file which is a superset of the features of Table 2 in the v2 paper. Details below.
- The `models` directory contains Python scripts that wrap the designated models. The model scripts are responsible for:
    + Running task rubrics (questions/answers) given a configuration that controls `temperature`, `top_p/k` etc..
    + Applying a rewrite function that rewrites questions before question answering.
- The `tasks` directory contains scripts that wrap access to the 8 tasks used in the evaluation. They wrap Hugging Face datasets and are responsible for: 
    + Generating training examples for few shot runs
    + Translation of multiple choice answers to letter based multiple choice.
    + Providing functions that return:
        - `raw_fn`: Raw LLM response used for total agreement rate raw (TARr)
        - `answer_fn`: Parsed answer from raw output for total agreement rates answer (TARa).
        - `correct_fn`: Whether the answer was correct or not for accuracy reporting.

## 3. Setting up Python

It is strongly recommended that you set up a virtual environment to evaluate or run experiments. There are many ways to achieve this which are operating system dependent, the way we do it is using the command line is:

- Install miniconda (https://docs.anaconda.com/miniconda/)
- `conda create -n llms python=3.11`
- `conda activate llms`

We are only using miniconda to manage our environment, we will use `pip` to install packages. 

- `pip install -r requirements.txt`

Now you should have a functioning environment to use. 

### 3.1 Setup to run evaluation

Run the following test to be sure the environment functions properly, note that these tests don't use external credentials. 

```
pytest tests/test_evaluate.py
pytest tests/test_tasks.py 
pytest tests/test_helper_functions.py 
```

## 3.2 Setup to run models

If you only want to evaluate existing output then you can ignore setting up the models and go to 4.  

Our experiments used hosted models which require credentials that you supply to run.
For example the `models/gpt-4o.py` script has the configuration:

```
AzureOpenAI(
                azure_endpoint=os.environ["AZURE_ENDPOINT_GPT_4_0"],
                api_key=os.environ["OPENAI_GPT4_KEY"],
                api_version="2024-04-01-preview",
                azure_deployment="AppliedAI-gpt-4o",
            )

```
You will have to set environment variables for `AZURE_ENDPOINT_GPT_4_0`, `OPENAI_GPT4_KEY` in the standard fashion of your operating system or you can put a
`.env` file in your home directory in the format:

```
AZURE_ENDPOINT_GPT_4_0 = "https://<your endpoint here>.openai.azure.com"
OPENAI_GPT4_KEY="<your key here>"
```
As you run additional models you will likely have to provide specific credentials and endpoints. 

WARNING: It is very easy to mistakenly push keys to repos so we strongly recommend not hard coding credentials into model scripts or putting a `.env` file in the repo directory--although there is a `.gitignore` that will not act upon a `.env` file. 

To verify that your credentials work there are tests for GPT-4o and GPT-3.5 Turbo. 
Add to these tests for models you wish to run. Below we test the individual LLMs:

- `pytest tests/test_models.py::test_gpt_4o` 
- `pytest tests/test_models.py::test_gpt_35_turbo`

## 4. Evaluate with `evaluate.py`

### 4.1 Extract previous run data as supplied in repo
This repo contains raw output in v2 data in `experiments/v2/runs.tgz` and v3 data in `experiments/v3/runs.tgz`. The v3 data has more runs and covers the same basic experiments as the v2 data so we use that. Steps to process:

- Uncompress and unarchive either via your operating system or from the linux style command line:
    ```
    > cd experiments/v3
    > tar -xfzf runs.tgz
    ```
- This will create a directory `runs` with the following structure:

    ```
    > cd runs/
    > ls
    gpt-35-turbo-0.0-college_mathematics-0_2024-12-28_17-04-16/
    gpt-35-turbo-0.0-college_mathematics-few_2024-12-28_16-42-30/
    ```

- The directory name convention is "<model>-<top_p_k>-<task>-<0 or few>-<date run started>"
Inside each directory are the .csv files from each run:

    `gpt-35-turbo-0.0-college_mathematics-0_2024-12-28_17-04-16/gpt-35-turbo-0.0-college_mathematics-0-0.csv`

    The same naming convention is applied to the file with the addition of a "-<[0-9]>.csv". For v3 data, there will be 10 runs numbered from 0-9. 

### 4.2 Run evaluation.py

The evaluation script is run by specifying the root of the directory where .csv files exist as supplied to the -d flag. It will recursively seek all the .csv files and attempt to evaluate them:

```
python evaluate.py -d experiments/v3/runs/
```

There is a mass of output typically generated that reports on answer parsing failures. Rerunning a single experiment directory makes this a bit more manageable to describe. 

```
> python evaluate.py -d experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08

Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0
.0-professional_accounting-0-8.csv
Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0
.0-professional_accounting-0-9.csv
Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0
.0-professional_accounting-0-1.csv
Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0.0-professional_accounting-0-0.csv
Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0.0-professional_accounting-0-2.csv
Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0.0-professional_accounting-0-3.csv
Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0.0-professional_accounting-0-7.csv
Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0.0-professional_accounting-0-6.csv
Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0.0-professional_accounting-0-4.csv
Trying experiments/v3/runs/gpt-4o-0.0-professional_accounting-0-2024-12-14_14-31-08/gpt-4o-0.0-professional_accounting-0-5.csv

```
Above are all the .csv files read in. 

The next line shows the parsed data from the .csv file that indicates what experiment is being run and the corresponding configuration--this should match the relevant portions of the file name--note that the v3 data filenames are for human consumption only, relevant information is pulled from the csv file. The v2 data does pull information from the file name, see the `evaluate.py::load_runs()` function for what is going on there:

```
gpt-4o {'temperature': 0.0, 'seed': 12, 'top_p_k': 0.0} professional_accounting {'prompt_type': 'v2', 'shots': 0}

```

Next in the output are problems encountered while parsing the LLM return strings for the answers. All our tasks are multiple choice but the LLMs often return varied answers that need to be addressed--look for a follow-on paper for answer parsing. 

The form of a failed parse report is the count of failed parses, here '0' because computer scientists count from 0, with a description of the failure: `Blown UP found gpt-4o professional_accounting 15 0` which is model 'gpt-4o' from rubric (question/answer pair from the task) #15 from run #0. 

```
------answer issue-----0---
Blown UP found gpt-4o professional_accounting 15 0
```
Blown UP (uniqueness presupposition) means that more than one answer was found.

Next we see what the LLM answered with:

```

Response: The type of audit evidence that provides the least assurance of reliability 
is typically the one that is generated internally by the client, as opposed to being 
obtained from an independent external source. In this case, option (B) "Prenumbered 
receiving reports completed by the client’s employees" is the least reliable because it 
is prepared internally and is subject to the client's internal controls and potential 
biases. 

In contrast, options (A), (C), and (D) involve evidence that is either obtained from 
external sources or involves third-party verification, which generally provides higher 
assurance of reliability.
```
So the answer parser is challenged to figure out what the answer is and fails. 
The rubric itself (question and answer) is supplied next to help figure out what is going on:

```Rubric: 
{'input': 'Which of the following types of audit evidence provides the least 
assurance of reliability?\n(A) Receivable confirmations received from the client’s 
customers.. (B) Prenumbered receiving reports completed by the client’s employees.. (C) 
Prior months’ bank statements obtained from the client.. (D) Municipal property tax bills 
prepared in the client’s name.. ', 
'target': '(B)'}
```
The answer for this data is the target, or `{'target': '(B)'}`. 

Next is a report on how many times the same exact LLM response happens so it does for runs #1, and #2. 

```

Repeat 1 for rubric id 15
Repeat 2 for rubric id 15

```
The other 8 runs for this rubric did not have parsing issues. 

The last line of output summarized the overall parser performance:

`19 rubrics had parsing problems for 76 task x rubrics for 2820 total evaluations`

The line indicates that 19 rubrics out of 282 rubrics had at least one parsing failure and the total count of parsing failures is 76 out of 2820 total runs. The counts and reporting will span all runs under the directory designated by the -d flag. For example:

```
> python evaluate.py -d experiments/v3/runs/
.....
264 rubrics had parsing problems for 2,003 task x rubrics for 66,280 total evaluations
```

We don't worry too much about missed parses if we are in the range of expected performance for the task since we are more interested in the stability of the raw LLM results but it is good to know and we are pursuing related research on robust answer parsing.


### 4.3 View `evaluation_output.csv` 

Open `evaluation_output.csv` in your favorite spreadsheet. The column labels are:


- model: Model name from 'model' column of .csv
- model_config: Configuration pulled from column 'model_config' of .csv
- task: Task name pulled from 'task' of .csv
- task_config: Configuration pulled from 'task_config' of .csv
- TACr: Total agreement (Python string equivalence) count for raw data across @N runs
- TARr: Total agreement rate raw, TACr/num_questions
- correct_count_per_run: List @N long of correct answers per run
- correct_pct_per_run: Percent correct per run
- num_questions: How many rubrics in task
- N: @N number of repeat runs
- best_possible_count: Count question correct if it was correctly answered on any run
- best_possible_accuracy: best_possible_count/N
- worst_possible_count: Count question incorrect if it was incorrectly answered on any run
- worst_possible_accuracy: worst_possible_count/N
- spread: best_possible_accuracy - worst_possible_accuracy
- bootstrap_counts: Sample with replacement 10 times and count correct
- bootstrap_pcts: boostrap_counts/num_questions
- date: Date of run

### 4.4 Available data in repository

The `experiments` directory contains all released runs for our experiments. The
contents must be uncompressed. On OSX the command `tar -xvzf runs.tgz` executed in the terminal in `experiments/v3/` directory. Your operating system may have other ways of uncompressing the file. Once uncompressed you can run `python evaluate -d experiments/v3` and once done view `stability_eval.csv`. 

The contents of `experiments` are as follows:

- `experiments/v2`: Contains most of the data used in the v2 paper. The data has to be evaluated with the old data flag, e.g., `python evaluate.py -eo -d experiments/v2`.
- 

## 5. Generate new runs

Running your own configurations/experiments requires that you pass the tests for the LLMs you want to use in section 3.1 and 3.2. Once the tasks/models are working then it should be trivial to run with different configurations. You are strongly encouraged to use the existing code base to run and evaluate to maintain consistency with other experiments. 

Three basic entry points exist to run data:
- Command line: `python run_experiment.py -m gpt-4o -mc '{"temperature":0.0, "seed": 12, "top_p_k": 0.0}' -t navigate -tc '{"prompt_type": "v2", "shots": 0}' -n 2 -l 3 -et` wraps command line configuration options and calls below `run(....)`
- Function call to `run_experiment.py::run(...)` which does file system checks, reads/writes to disk based on parameterization and runs one task against one model N times by calling `run_model()`. The code in `run_table_2.py` shows how to use this function to run a bigger experiment and not re-run experiments that don't need to be run. 
- Function call to `run_experiment.py::run_model(...)` which is where a single task is run N times against one task and output written to specified file. The script `run_from_experiment_output.py` uses this function to rerun experiments from the output .csv alone as a proof of reproducibility if the Hugging Face data source goes away.

### 5.1 Creating different models/ or tasks/

 If you are adding your own model then use as a template  an existing model and implement it. The same applies if you wish to add your own task, use an existing task as a template and add it to the `tasks` directory and unit test it. 

 There are `models/README.md` and `tasks/README.md` to give more information about the design and what config options like `{"prompt": "v2"}` mean. 

### 5.2 Running from command line `run_experiment.py`
 
The command line gives fairly good access different ways of running the evaluation without writing code. There is fairly decent help if you run `python run_experiment.py -h`:

```
usage: python run_experiment.py -m gpt-4o -mc '{"temperature":0.0, "seed": 12, "top_p_k": 0.0}' -t navigate -tc '{"prompt_type": "v2", "shots": 0}' -n 2 -l 3 -et

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Name of module in models/
  -mc MODEL_CONFIG, --model_config MODEL_CONFIG
                        Configuration for model
  -t TASK, --task TASK  Name of task module in tasks/
  -tc TASK_CONFIG, --task_config TASK_CONFIG
                        Configuration for task
  -n NUM_RUNS, --num_runs NUM_RUNS
                        Number of runs to execute
  -d OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Where to write output files, will create all directories
  -l LIMIT_NUM_RUBRICS, --limit_num_rubrics LIMIT_NUM_RUBRICS
  -et, --use_earliest_time_stamp
                        Creates time stamp: 0001-01-01_00-00-00. Will overwrite previous
                        run.

```

The usage example runs standard configurations for the model/task and runs 3 rubrics from the navigate task two times. Output is:

```
python run_experiment.py -m gpt-4o -mc '{"temperature":0.0, "seed": 12, "top_p_k": 0.0}' -t navigate -tc '{"prompt_type": "v2", "shots": 0}' -n 2 -l 3 -et
No existing file, running gpt-4o-0.0-navigate-0
Model loaded
Limiting test to first 3 rubrics
Data loaded
Making output directories if necessary local_runs/gpt-4o-0.0-navigate-0_0001-01-01_00-00-00
File system ready for run
Running 0
100%|█████████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.15s/it]
*** Wrote local_runs/gpt-4o-0.0-navigate-0_0001-01-01_00-00-00/gpt-4o-0.0-navigate-0-0.csv
Running 1
100%|█████████████████████████████████████████████████████████| 3/3 [00:09<00:00,  3.07s/it]
*** Wrote local_runs/gpt-4o-0.0-navigate-0_0001-01-01_00-00-00/gpt-4o-0.0-navigate-0-1.csv
Run successful, run `python evaluate.py -d local_runs/gpt-4o-0.0-navigate-0_0001-01-01_00-00-00

```

The last line gives you the command to run the `evaluation.py` script. Note that the `run_experiment.py` will not run if there is the same configuration in the destination directory as determined by the file name independent of the timestamp. Run with the command line `-et` option to overwrite previous configuration equivalent runs or just delete the blocking directory from the output directory.

### 5.3 Running from output .csv files

For reproducibility reasons there is sufficient data in the output files to rerun the experiments. 

- `run_from_experiment_output.py` takes existing experiment output and will rerun from the configuration information in the output files. The only degree of freedom is the LLM used to run the task. If the hosted LLMs change then the experiments will not be reproducible.  
