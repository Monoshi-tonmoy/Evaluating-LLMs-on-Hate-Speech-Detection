# Evaluation of Large Language Models on Hate Speech Dataset

## File Structure
```bash
├── data
│   ├── dataset.jsonl
│   ├── Master1.csv
│   └── Processing.ipynb
├── results
│   └── llama2-7b
│       └── translated
│           └── results.jsonl
└── scripts
    ├── generate_prompt
    │   ├── generate_translated_prompting.py
    │   ├── __init__.py
    │   └── utils.py
    ├── global_variables.py
    ├── __init__.py
    ├── main.py
    ├── models.py
    └── utils.py
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
The results will be saved in results/llama2-7b/translated/results.jsonl with the following fields:
 - id, Comment, Translated_Comment, Hate Speech, model, prompt_mode, prompt_template, response_data. 
```
