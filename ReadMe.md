# Evaluation of Large Language Models on Hate Speech Dataset

## File Structure
```bash
.
├── codellama34_results
│   └── codellama7
│       └── translated
│           └── results.jsonl
├── data
│   ├── Bengali_data
│   │   └── results.jsonl
│   ├── Bengali_filtered_dataset.jsonl
│   ├── codellama_34b_Bengali.jsonl
│   ├── dataset.jsonl
│   ├── gpt3.5_Bengali.jsonl
│   ├── Master1.csv
│   ├── Processing.ipynb
│   ├── sample_adv_dataset.jsonl
│   ├── sampled_dataset_500.jsonl
│   ├── sampled_dataset.jsonl
│   └── sample_ori_adv_dataset.jsonl
├── gpt_results
│   └── codellama7
│       └── translated
│           └── results.jsonl
├── ReadMe.md
├── requirements.txt
├── results
│   ├── codellama7
│   │   └── translated
│   │       └── results.jsonl
│   ├── deepseek6.7
│   │   └── translated
│   │       └── results.jsonl
│   └── llama
│       └── translated
│           └── results.jsonl
├── results_adv
│   ├── codellama7
│   │   └── adversarial
│   │       └── results.jsonl
│   ├── deepseek6.7
│   │   └── adversarial
│   │       └── results.jsonl
│   └── llama
│       ├── adversarial
│       │   └── results.jsonl
│       └── translated
├── results_org
│   ├── codellama7
│   │   ├── adversarial
│   │   │   └── results.jsonl
│   │   └── translated
│   │       └── results.jsonl
│   ├── deepseek6.7
│   │   └── adversarial
│   │       └── results.jsonl
│   └── llama
│       ├── adversarial
│       │   └── results.jsonl
│       └── translated
└── scripts
    ├── Bengali_Analysis.ipynb
    ├── generate_prompt
    │   ├── generate_adversarial_prompting.py
    │   ├── generate_bd_translation_prompting.py
    │   ├── generate_original_prompting.py
    │   ├── generate_translated_prompting.py
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── generate_adversarial_prompting.cpython-310.pyc
    │   │   ├── generate_bd_translation_prompting.cpython-310.pyc
    │   │   ├── generate_translated_prompting.cpython-310.pyc
    │   │   └── __init__.cpython-310.pyc
    │   └── utils.py
    ├── global_variables.py
    ├── __init__.py
    ├── main.py
    ├── model_metrics.csv
    ├── models.py
    ├── parser.ipynb
    ├── performance_comparison.png
    ├── __pycache__
    │   ├── global_variables.cpython-310.pyc
    │   ├── models.cpython-310.pyc
    │   └── utils.cpython-310.pyc
    ├── tp_vs_location.png
    └── utils.py
```

# Dataset
```bash
You can find all the datasets in the data folder
 -- dataset.jsonl contains all 10k samples
 -- sampled_dataset.jsonl contains evaluation samples
 -- sampled_original_dataset.jsonl is the English original hatespeech dataset
 -- sample_ori_adv_dataset.jsonl is the English adversarial hatespeech dataset
 -- gpt3.5_Bengali.jsonl is the gpt translated dataset
 -- codellama_34b_Bengali.jsonl is the codellama-34b translated dataset
```

# How to run the models
You will find the different model names and their pre-tuned checkpoint links in "scripts/global_variables.py" file.

Run the evaluation with following command:
```bash
python scripts/main.py --model <model_name> --dataset <dataset_name> --template <template_version> --mode <mode> --n <num_samples>
--model: The model to use (e.g., llama).
--dataset: The dataset file (e.g., dataset for data/dataset.jsonl).
--template: The prompt template version (e.g., translated).
--mode: The mode for prompt generation (e.g., translated).
--n: The number of samples to evaluate
```


# Results
```bash
The results of hate speech evaluation (Section V, table I, II) will be saved in results/{model_name}/{mode}/results.jsonl with the following fields:
 - id, Comment, Translated_Comment, Hate Speech, model, prompt_mode, prompt_template, response_data. 
 The results of adversarial analysis (Section V, table III) will be saved in results_adv and results_org with the following fields:
 - id, Comment, Translated_Comment, Hate Speech, model, prompt_mode, prompt_template, response_data. 
 The results of codellama34b and gpt3.5 analysis (Section V, table IV) will be saved in codellama34_results and gpt_results with the following fields:
 - id, Comment, Translated_Comment, Hate Speech, model, prompt_mode, prompt_template, response_data. 
```

# Parsing and metrics calculation
```bash
Output parsing, metrics calcuation and figure generation scripts are availale in scripts/parser.ipynb file
```

