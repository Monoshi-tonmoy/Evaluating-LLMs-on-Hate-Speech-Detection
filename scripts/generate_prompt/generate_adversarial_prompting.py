import random


def make_direct_output_prompt():
    return f"""You will be given a comment of English Language. Based on the given comment, you need to predict whether it contains hate speech or not. If the comment contains hate speech, provide your prediction inside [ANSWER][/ANSWER] tags. Below are some examples:

[COMMENT]
That guy is an asshole.
[/COMMENT]
[QUESTION]
Does this comment contain hate speech?
[/QUESTION]
[ANSWER]
Yes
[/ANSWER]

[COMMENT]
She looks pretty.
[/COMMENT]
[QUESTION]
Does this comment contain hate speech?
[/QUESTION]
[ANSWER]
No
[/ANSWER]

"""


def generate_adversarial_prompt(args, data_entry):
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


def generate_adversarial_prompt_template_vadversarial(args, query_ex=None):
    return generate_adversarial_prompt(args, query_ex)