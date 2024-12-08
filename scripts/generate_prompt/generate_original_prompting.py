import random


def make_direct_output_prompt():
    return f"""You will be given a comment from one of the following regions: Arab, Bangladesh, India, China, or Russia/Ukraine. Based on the given comment, you need to predict whether it contains hate speech or not. If the comment contains hate speech, provide your prediction inside [ANSWER][/ANSWER] tags. Below are some examples:

[COMMENT]
أنت عديم الفائدة ولا تستحق العيش
[/COMMENT]
[QUESTION]
Does this comment contain hate speech?
[/QUESTION]
[ANSWER]
Yes
[/ANSWER]

[COMMENT]
তুই একটা অকর্মা, তোর সমাজে কোন জায়গা নেই।
[COMMENT]
[QUESTION]
Does this comment contain hate speech?
[/QUESTION]
[ANSWER]
Yes
[/ANSWER]


[COMMENT]
आप बहुत मेहनती हैं और हमेशा अच्छा काम करते हैं।
[COMMENT]
[QUESTION]
Does this comment contain hate speech?
[/QUESTION]
[ANSWER]
No
[/ANSWER]
"""


def generate_original_prompt(args, data_entry):
    source_code = data_entry["Comment"] 

    prompt = ""
    query_prompt = ""

    prompt = make_direct_output_prompt()
    
    query_prompt = f"""
[COMMENT]
{source_code}
[/COMMENT]
[QUESTION]
Does the comment contain any hatespeech?
[/QUESTION]
"""
        
    return prompt + "\n\n\n" + query_prompt


def generate_original_prompt_template_vtranslated(args, query_ex=None):
    return generate_original_prompt(args, query_ex)

