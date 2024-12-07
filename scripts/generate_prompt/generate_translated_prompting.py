import random


def make_direct_output_prompt():
    return f"""You will be given a translated comment from one of the following regions: Arab, Bangladesh, India, China, or Russia/Ukraine. Based on the given translated comment, you need to predict whether it contains hate speech or not. If the comment contains hate speech, provide your prediction inside [ANSWER][/ANSWER] tags. Additionally, identify the geographical region of the comment and provide your prediction in [LOCATION][\LOCATION] tags. Below are some examples:

[TRANSLATED_COMMENT]
Allah is great.
[/TRANSLATED_COMMENT]
[QUESTION]
Does this comment contain hate speech? And which region is the comment from?
[/QUESTION]
[ANSWER]
No
[/ANSWER]
[LOCATION]
Arab
[/LOCATION]

[TRANSLATED_COMMENT]
Papan, you are a gandu.
[/TRANSLATED_COMMENT]
[QUESTION]
Does this comment contain hate speech? And which region is the comment from?
[/QUESTION]
[ANSWER]
Yes
[/ANSWER]
[LOCATION]
Bangladesh
[/LOCATION]


[TRANSLATED_COMMENT]
You are a behenchod and should stay out of this.
[/TRANSLATED_COMMENT]
[QUESTION]
Does this comment contain hate speech? And which region is the comment from?
[/QUESTION]
[ANSWER]
Yes
[/ANSWER]
[LOCATION]
India
[/LOCATION]
"""


def generate_translated_prompt(args, data_entry):
    source_code = data_entry["Translated_Comment"] 

    prompt = ""
    query_prompt = ""

    prompt = make_direct_output_prompt()
    
    query_prompt = f"""
[TRANSLATED_COMMENT]
{source_code}
[/TRANSLATED_COMMENT]
[QUESTION]
Does the comment contain any hatespeech? And The comment is based on which location?
[/QUESTION]
"""
        
    return prompt + "\n\n\n" + query_prompt


def generate_translated_prompt_template_vtranslated(args, query_ex=None):
    return generate_translated_prompt(args, query_ex)

