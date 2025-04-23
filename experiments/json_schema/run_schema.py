import os
import sys
sys.path.append(os.path.join(os.getcwd()))
import helper_functions
import run_experiment
from datetime import datetime, date, MINYEAR


schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "experiment",
            "schema": {
                "type": "object",
                "properties": {
                                "Answer": {
                                  "type": "string",
                                  "enum" : ["Yes", "No"]
                                }
                              },
                              "required": [
                                "Answer"
                              ],
                              "additionalProperties": False
            },
            "strict": True
        }
}


json_schema_Yes_No = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "Answer": {
      "type": "string",
      "enum" : ["Yes", "No"]
    }
  },
  "required": [
    "Answer"
  ]
}



json_schema_prompt_Yes_No = """
Please answer the following question adhering to these format instructions:
The output should be formatted as a JSON instance that conforms to the JSON schema below.
 
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "Answer": {
      "type": "string",
      "enum" : ["Yes", "No"]
    }
  },
  "required": [
    "Answer"
  ]
}

The output {"Answer": "Yes"} is a well-formatted instance of the schema, the output {"Answer": "E"} is not well-formatted. A string answer like "The correct answer is Yes" is not well-formatted.

The question is: 


"""

json_schema_prompt_A_D = """
Please answer the following question adhering to these format instructions:
The output should be formatted as a JSON instance that conforms to the JSON schema below.
 
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "Answer": {
      "type": "string",
      "enum" : ["A", "B", "C", "D"]
    }
  },
  "required": [
    "Answer"
  ]
}

The output {"Answer": "A"} is a well-formatted instance of the schema, the output {"Answer": "E"} is not well-formatted. A string answer like "The correct answer is A" is not well-formatted.

The question is: 


"""

json_schema_prompt_A_K = """
Please answer the following question adhering to these format instructions:
The output should be formatted as a JSON instance that conforms to the JSON schema below.
 
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "Answer": {
      "type": "string",
      "enum" : ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    }
  },
  "required": [
    "Answer"
  ]
}

The output {"Answer": "A"} is a well-formatted instance of the schema, the output {"Answer": "L"} is not well-formatted. A string answer like "The correct answer is A" is not well-formatted.

The question is: 


"""


MODELS = ['gpt-4o', 'gpt-35-turbo', 'gpt-4o_OAI']
MODEL_CONFIGS = [{'temperature': 0.0, 'seed': 12, 'top_p_k': 0.0,
                 'prefix': ''}]

YES_NO_TASKS = ['navigate']

A_K_TASKS = ['geometric_shapes']

A_D_TASKS = ['professional_accounting',  'college_mathematics',
         'logical_deduction', 'ruin_names',        
          'high_school_european_history',
         'public_relations'
        ]

TASKS = YES_NO_TASKS + A_K_TASKS + A_D_TASKS


TASK_CONFIGS = [{'prompt_type': 'v2', 'shots': 'few'}]
                # 'prefix': '\n\n', 'suffix': '\n\n'}]
                #{'prompt_type': 'v2', 'shots': 0}]



experiments = helper_functions.experiment_setup(MODELS, MODEL_CONFIGS, 
                                                TASKS, TASK_CONFIGS)


for model, model_config, task, task_config in experiments:
    date = datetime.now()
    datetime_string = date.strftime("%Y-%m-%d_%H-%M-%S")
    if task in A_D_TASKS:
        model_config['prefix'] = json_schema_prompt_A_D
    elif task in YES_NO_TASKS:
        model_config['prefix'] = json_schema_prompt_Yes_No
    elif task in A_K_TASKS:
        model_config['prefix'] = json_schema_prompt_A_K
    else:
        print(f"No schema for {task}, skipping")
        continue
    run_args = {'output_directory': 'json_schema/runs/',
                'model': model,
                'model_config': model_config,
                'model_config_in_filename': model_config['temperature'],
                'task': task,
                'task_config': task_config,
                'task_config_in_filename': task_config['shots'],
                'num_runs': 5,
                'schema': model_config['prefix']
#                'limit_num_rubrics': 1
    }
#    datetime_string = '0001-01-01_00-00-00'
    val = run_experiment.run(run_args, datetime_string)
    if val == "Successfully run":
        break