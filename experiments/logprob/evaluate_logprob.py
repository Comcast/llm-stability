import pandas as pd
import glob
import os
import json
import sys
import importlib
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
import evaluate
from answer_parser import parse_paren_answers_with_weights 
import math
import plotnine as p9



files = glob.glob(os.path.join('runs', '**', '*.csv'), 
                        recursive=True)
data_df = pd.DataFrame()
for csv_file in files:
    csv_df = pd.read_csv(csv_file)
    print(f"Trying {csv_file}")
    for col in ['prompt', 'model_config', 'task_config', 'rubric','logprobs']:
        csv_df[col] = csv_df[col].apply(lambda cell: json.loads(cell))
    csv_df['file'] = csv_file
    data_df = pd.concat([data_df, csv_df], ignore_index=True)


#get answer's logprob
configs = evaluate.get_experiment_configs(data_df)
task_x_rubric = set()
total_evals = 0
disp_df = pd.DataFrame()
for model, model_config, task, task_config in configs:
    try:
        task_module = importlib.import_module(f'tasks.{task}')
    except ModuleNotFoundError:
        print(f'Need to add {f"tasks.{task}"}, skipping eval')
        continue
    results = pd.DataFrame()
    exp_df = data_df[(data_df['model'] == model)
                        & (data_df['model_config'] == model_config)
                        & (data_df['task'] == task)
                        & (data_df['task_config'] == task_config)]
    #print(f"{model} {model_config} {task} {task_config}")
    if len(exp_df.index) == 0: #may have combos with no data
        continue
    rubric_ids = exp_df['rubric_id'].unique()
    num_questions = len(rubric_ids)
    num_runs = max(exp_df['run']) + 1
    
    for id in rubric_ids:
        task_x_rubric.add(f"{task}x{id}")
        question_df = exp_df[exp_df['rubric_id'] == id]
        raw = set()
        answer = set()
        logprob = set()

        if not num_runs == len(question_df.index):
            print(f"{model}, {model_config}, {task}, {task_config}")
            error = f"runs not matching expected length, expected {num_runs}, got {len(question_df.index)} for {question_df['file'].to_list()}"
            raise IndexError(error)
        corrects_for_rubric = 0
        run_accum = [0] * num_runs
        prob_accum = []
        logprob_accum = []
        answers_accum = []
        response_token_logprob_accum = []
        response_token_avg_logprob_accum = []
        response_token_avg_prob_accum = []
        token_lp_accum = []
        
        for idx, row in question_df.iterrows(): # runs over question
            raw.add(task_module.raw_fn(row))
            total = 0
            tok_lp_list = []
            for pr in row['logprobs']:
                total += pr['logprob']
                tok_lp_list.append(pr['logprob'])
            token_lp_accum.append(tok_lp_list)
            avg_logprob = total/len(row['logprobs'])
            response_token_avg_logprob_accum.append(avg_logprob)
            response_token_avg_prob_accum.append(math.exp(avg_logprob))
            total_evals += 1
            if pd.isna(row['response']):
                errors.append(f"NaN response found {model} {task} {id}")
                prob_accum.append(None)
                logprob_accum.append(None)
                answers_accum.append('na')
                continue
            try:
                parsed_answer = task_module.answer_fn(row, task_config)
                if parsed_answer is None:
                    error = f"No answer found {model} {task} {id}"
                    #print(error)
                    #print(f"Response: {row['response']}")
                    prob_accum.append(None)
                    logprob_accum.append(None)
                    answers_accum.append(None)
                    continue # cannot be correct so continue
                else:
                    data_df.loc[idx,'parsed_answer'] = parsed_answer
                    answer.add(parsed_answer)
                    answer_2 = \
                        parse_paren_answers_with_weights([parsed_answer], row)
                    if answer_2 is None:
                        print("Unable to parse previously parsable answer")
                        print(f"Response: {row['response']}")
                    data_df.loc[idx,'logprob'] = answer_2[parsed_answer]
                    logprob_accum.append(answer_2[parsed_answer])
                    prob_accum.append(math.exp(answer_2[parsed_answer]))
                    answers_accum.append(parsed_answer)
            except LookupError as e: #LookupError is 
                answer.add(idx) # Blown UP is also a failure of TARa
                data_df.loc[idx,'parsed_answer'] = "Blown UP"
                error = f"Blown UP found {model} {task} {id} Answer: {e}"
                prob_accum.append(None)
                logprob_accum.append(None)
                answers_accum.append('BUP')
                continue # can't be correct so continue
        runs_df = pd.DataFrame({'answer prob': prob_accum,
                                'response prob': response_token_avg_prob_accum})
        runs_df['id'] = id
        runs_df['task'] = task
        runs_df['TARr'] = len(raw)/num_runs
        runs_df['model'] = model
        disp_df = pd.concat([disp_df, runs_df], ignore_index=True)
        # if len(raw) == 5:
        #     print(f"\n{len(raw)} raw results for {len(question_df.index)}")
        #     print(f"Answers are: {answers_accum}")
        #     print(f"Parsed answer prob: {prob_accum}")
        #     print(f"Avg tok prob entire response: {response_token_avg_prob_accum}")
        #     #print(f"Tok logprob: {token_lp_accum}")
        #print(logprob_accum)
    #print(disp_df)

disp_df = disp_df.dropna(subset=['answer prob', 'response prob'])

for model in disp_df['model'].unique():
    if model != 'llama3-8b':
        continue
    breakpoint()
    model_df = disp_df[disp_df['model'] == model]
    melted_df = model_df.melt(
        id_vars=['TARr'],
        value_vars=['answer prob', 'response prob'],
        var_name='prob_type',
        value_name='probability'
    )
    disp_df['TARr 2'] = disp_df['TARr'] + .01
    plot = (
        p9.ggplot(melted_df, p9.aes(x='TARr', y='probability', color='prob_type')) +
        p9.geom_point(position=p9.position_jitter(width=0.01, height=0), alpha=0.7) +
        #p9.theme_xkcd() +
        p9.labs(
            title='Per Rubric Answer/Response Probs TARr@5',
            x='TARr (jittered)',
            y='Avg Probability over Tokens',
            color='Type'
        )
    )
    plot.save("prob_v_TARr.png", width=6, height=4, dpi=300)
    #plot