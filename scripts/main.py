import argparse
import tqdm
from utils import *
from models import *
from generate_prompt import *
from global_variables import data_dir, models_ids, api_types, deprecated_models_ids, deprecated_api_types, result_dir, result_dir_org, result_dir_adv
import transformers
from pathlib import Path
import re


def write_to_jsonl(datapath, record):
    with open(datapath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
def get_result_directory(args):
    if args.dataset == "sample_ori_adv_dataset":
        rdir = f"{result_dir_org}/{args.model}/{args.template}"
    elif args.dataset == "sample_adv_dataset":
        rdir = f"{result_dir_adv}/{args.model}/{args.template}"
    elif args.dataset == "Bengali_Translation":
        rdir = f"{data_dir}/Bengali_data"
    elif args.dataset == "codellama_34b_Bengali":
        rdir = f"../codellama34_results/{args.model}/{args.template}"
    elif args.dataset == "gpt3.5_Bengali":
        rdir = f"../gpt_results/{args.model}/{args.template}"
    else:
        rdir = f"{result_dir}/{args.model}/{args.template}"
    return Path(rdir)


def get_model(args):
    args.model_id = models_ids[args.model]
    args.api_type = api_types[args.model]

    def warn_model_name():
        print("WARNING: Using deprecated model name. Try to use the updated model names...")
    if args.model in deprecated_models_ids or args.model in deprecated_api_types:
        warn_model_name()

    args.model_ = model = Model(model_id=models_ids[args.model], api_type=api_types[args.model])
    return model

def get_prompt_function(args):
    return eval(f"generate_{args.mode}_prompt_template_v{args.template}")

def get_answer(prompt, args):
    args.model_.ensure_model_initialized()
    responses = []
    responses.append(args.model_.query(prompt, return_metadata=True))
    return responses

def add_arguments(parser):
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--template", type=str)
    parser.add_argument("--mode", type= str)
    parser.add_argument("--n", type= int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    args.model_ = get_model(args)
    
    if args.dataset == "Bengali_Translation":
        args.data = read_jsonl(f"{data_dir}/Bengali_filtered_dataset.jsonl")
    else:
        args.data = read_jsonl(f"{data_dir}/{args.dataset}.jsonl")
    get_prompt = get_prompt_function(args)
    

    results = []

    args.result_directory = get_result_directory(args)
    
    result_directory = args.result_directory/"results.jsonl"


    result_directory.parent.mkdir(exist_ok=True, parents=True)

    for i, ex in enumerate(tqdm.tqdm(args.data[:args.n])):
        prompt = get_prompt(args, query_ex=ex)
        response = get_answer(prompt, args)
        
        if args.dataset == "Bengali_Translation":
            translated_english = response[0]['output_without_input']
            match = re.search(r'\[TRANSLATION\]\s*(.*?)\s*\[\/TRANSLATION\\?\]', translated_english, re.DOTALL)
            enlgish_sentence = match.group(1) if match else None
            write_to_jsonl(result_directory, {
                "id": ex["idx"],
                "Comment": ex["Comment"],
                "Translated_Comment": enlgish_sentence,
                "Hate Speech": ex["Hate Speech"],
                "Geography": ex["Geography"]
            })
        elif args.dataset == "codellama_34b_Bengali":
            write_to_jsonl(result_directory, {"id": ex["id"], "Comment": ex["Comment"], "Translated_Comment": ex["Translated_Comment"], "Hate Speech": ex["Geography"], "label": ex["Hate Speech"], "model": args.model, "prompt_mode": args.mode, "prompt_template": args.template, "response_data": response[0]})
        else:    
            write_to_jsonl(result_directory, {"id": ex["idx"], "Comment": ex["Comment"], "Translated_Comment": ex["Translated_Comment"], "Hate Speech": ex["Geography"], "label": ex["Hate Speech"], "model": args.model, "prompt_mode": args.mode, "prompt_template": args.template, "response_data": response[0]})




