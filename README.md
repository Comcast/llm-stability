# LLM Stability: A detailed analysis with some surprises


Berk Atil (bka5352@psu.edu), Alexa Chittams, Lisheng Fu, Ferhan Ture, Lixinyu Xu, Breck Baldwin (breck_baldwin@comcast.com)

v1.0 Submitted July 26, 2024 ArXiv released August 6, 2024: https://arxiv.org/abs/2408.04667

v2.0 ...

## Supporting software and data

The repo contains source code used to query the LLMs, `main.py`, the parsing code to extract answers `post_process_answers.py` and the scoring software `calculate_statistics.py`. Note that this project was started without the intention of releasing source and data so there may be small discrepencies with versions of the paper particularly in the parsing out of answers. The variations are minor and should not impact the analysis in the paper. 

The data from the experiment runs reported in v1.0 (few-shot) and the additional runs for various alternate configurations (few-shot, 0-shot and fine-tuned) in v2.0 are in `release_data.tgz` which can be uncompressed with the command line utility `tar -xvzf release_data.tgz` on linux/osx or which results in a folder `release_data`. GUI window managers also typcially can extract the compressed data as well by double clicking on the file. It has sub folders
* `release_data/few_shot` which are the results from the few shot runs repeated 5 times. This contains the preliminary data that setup the paper. 
* `release_data/20_run` which is the 20 run example used to help characterize the shape of score variations.
* `release_data/0-shot` is a collection of 5 run evaluations with no training examples provided.
* `release_data/fine-tuned-few-shot` contain the results of fine tuning several models on some of the tasks. 

The above data represents 520 runs ranging from 100 to 250 questions each and as such there remains a great deal of potential analysis over the experiments we ran but did not conduct. We release this data in the hopes of others taking advantage of a rather expensive and time consuming effort. 


## Setup

Our LLMs were hosted on a variety of services that are specific to our company. The function `configure_model` in `helper_functions.py` will have to be recreated for your local infrastructure. 

In addition there is a `requirements.txt` file that details the modules needed to run the software. Typically one creates a Python virtual environment and then run `pip install -r requirements.txt` for installation. 

## Explanations for the files

The files below are offered more in the hopes of clarity about what was done exactly for the purposes of reproducability than serving as a foundation for related work. The processes are just not that complex and our implementation not intended to live beyond the needs of the experiments but we have tried to make the code and steps we took as clear as possible. 

- `main.py`: This is the main file to run the LLMs. It has 4 named parameters:
  - `task`: which is to specify a task which can be any of the followings: mmlu_professional_accounting,  bbh_geometric_shapes, bbh_logical_deduction_three_objects, bbh_navigate, bbh_ruin_names, mmlu_college_mathematics, mmlu_high_school_european_history
  - `model`: the name of the model such as gpt-3.5-turbo, gpt-4o, or llama-3-70b
  - `num_few_shot`: number of few shot examples being used, they are set in the paper as follows for the few shot examples : bbh:3, mmlu:5. There are also 0-shot runs. 
  - `experiment_run`: this is to keep track of different runs, 5 for most tasks with some runs at 20 to characterize the shape of the accuracy distribution.
  - Example call: `python main.py  --model gpt-3.5-turbo --task mmlu_professional_accounting --experiment_run 0 --num_fewshot 4`. Running this will create a file `gpt-3.5-turbo_mmlu_professional_accounting_0.csv` that is a sibling to `main.py`. Note that the `num_fewshot` parameter was not included in the output filename convention and was added by hand later. We suggest collecting runs into an appropriately named folder as done in our `release_data` folder.
- `helper_functions.py`: Helper functions here that are being used in other files.
- `postprocess_responses.py`: We encountered parsing complexities in extracting answers.  This file after a run is done to get rid of some of the parsing issues but sometimes manual checking is inevitable. 
- `calculate_statistics.py`: This calculates the reported metrics after all parsings are done. An example call is: `python calculate_statistics.py release_data/few_shot fewshot_scores` which creates the indicated folder if needed and writes the files:
    + `fewshot_scores/detail_accuracy.csv`: Parsed outputs for each model/task/run
    + `fewshot_scores/low_med_high.csv` Accuracy, low, median, high
    + `fewshot_scores/TARa.csv` Total agreement rate answer results
    + `fewshot_scores/TARr.csv` Total agreement rate raw results
