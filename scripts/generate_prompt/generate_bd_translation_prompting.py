def make_direct_output_prompt():
    return f"""You will be given a Bengali Comment, and your task is to translate the Bengali Comment to English Comment without changing the Sentiment of the original language. Give the English Translation inside [TRANSLATION] [/TRANSLATION] tags, don't print any extra information.\n
Here are some examples\n:
[BENGALI_COMMENT]
তুই তো একটা অকর্মণ্য গাধা। তোর মতো মানুষের এই দেশে থাকা উচিত না।
[/BENGALI_COMMENT]
Translate the Benglai comment to sentiment preserving English Comment.
[TRANSLATION]
You are such a useless donkey. People like you shouldn't even be in this country.
[/TRANSLATION]

[BENGALI_COMMENT]
তুমি কি কাজটা ঠিকমতো করতে পারছো না? উন্নতি করার সুযোগ তো অনেক আছে।
[/BENGALI_COMMENT]
Translate the Benglai comment to sentiment preserving English Comment.
[TRANSLATION]
Aren’t you able to do the task properly? There’s plenty of room for improvement.
[/TRANSLATION]
"""


def generate_translated_prompt(args, data_entry):
    source_comment = data_entry["Comment"] 

    prompt = ""
    query_prompt = ""

    prompt = make_direct_output_prompt()
    
    query_prompt = f"""
[BENGALI_COMMENT]
{source_comment}
[/BENGALI_COMMENT]
Translate the Benglai comment to sentiment preserving English Comment.
"""   
    return prompt + "\n\n\n" + query_prompt


def generate_bd_translation_prompt_template_vbd_translation(args, query_ex=None):
    return generate_translated_prompt(args, query_ex)
