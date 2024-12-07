data_dir = "../data"
result_dir = "../results"

models_ids = {
    # OpenAI
    "gpt3.5": "gpt-3.5-turbo-0613",
    "gpt4": "gpt-4-0613",
    "gpt3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt4-turbo": "gpt-4-0125-preview",

    # Google
    "gemini": "gemini-pro",
    "palm2": "chat-bison-1",

    # Meta
    "llama2": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "llama2-noinstruct": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-noinstruct": "meta-llama/Llama-2-7b-hf",
    "llama2-70b-noinstruct": "meta-llama/Llama-2-70b-hf",
    "codellama": "codellama/CodeLlama-34b-Instruct-hf",
    "codellama13": "codellama/CodeLlama-13b-Instruct-hf",
    "codellama7": "codellama/CodeLlama-7b-Instruct-hf",
    "codellama-noinstruct": "codellama/CodeLlama-34b-hf",
    "codellama13-noinstruct": "codellama/CodeLlama-13b-hf",
    "codellama7-noinstruct": "codellama/CodeLlama-7b-hf",
    "gemma2": "google/gemma-2-2b-it",
    "llama3": "meta-llama/Llama-3.1-8B",
    "semcoder": "semcoder/semcoder",
    "semcoder-s": "semcoder/semcoder_s_1030",

    "deepseek33": "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek6.7": "deepseek-ai/deepseek-coder-6.7b-instruct",

    # Mistral
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",

    # StarCoder
    "starcoder-ta": "bigcode/starcoder",
    "starcoder2-15b": "bigcode/starcoder2-15b",

    # WizardCoder
    "wizardcoder": "WizardLM/WizardCoder-15B-V1.0", # 15B is the largest model available to us because 34B is trained for Python

    # Magicoder
    "magicoder-cl": "ise-uiuc/Magicoder-CL-7B",
    "magicoder-ds": "ise-uiuc/Magicoder-DS-6.7B",
    "magicoder-s-cl": "ise-uiuc/Magicoder-S-CL-7B",
    "magicoder-s-ds": "ise-uiuc/Magicoder-S-DS-6.7B",
    "magicoder": "ise-uiuc/Magicoder-S-CL-7B",

    "starchat-beta": "HuggingFaceH4/starchat-beta",
}

legacy_to_ollama = {
    # Meta
    "meta-llama/Llama-2-13b-chat-hf": "llama2:13b-chat",
    "meta-llama/Llama-2-7b-chat-hf": "llama2:7b-chat",
    "meta-llama/Llama-2-70b-chat-hf": "llama2:70b-chat",
    "meta-llama/Llama-2-13b-hf": "llama2:13b",
    "meta-llama/Llama-2-7b-hf": "llama2:7b",
    "meta-llama/Llama-2-70b-hf": "llama2:70b",
    "codellama/CodeLlama-34b-Instruct-hf": "codellama:34b-instruct",
    "codellama/CodeLlama-13b-Instruct-hf": "codellama:13b-instruct",
    "codellama/CodeLlama-7b-Instruct-hf": "codellama:7b-instruct",
    "codellama/CodeLlama-34b-hf": "codellama:34b",
    "codellama/CodeLlama-13b-hf": "codellama:13b",
    "codellama/CodeLlama-7b-hf": "codellama:7b",
    "deepseek-ai/deepseek-coder-33b-instruct": "deepseek-coder:33b-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "deepseek-coder:6.7b-instruct",

    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral:7b-instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mixtral:instruct",

    # StarCoder
    "bigcode/starcoder": "bstee615/starcoder:15b-ta",
    "HuggingFaceH4/starchat-beta": "bstee615/starchat-beta:15b",
    # "HuggingFaceH4/starchat-beta": "sqs/starchat:beta-q4_0",

    # WizardCoder
    "WizardLM/WizardCoder-15B-V1.0": "wizardcoder:33b", # 15B is the largest model available to us because 34B is trained for Python

    # Magicoder
    "ise-uiuc/Magicoder-S-CL-7B": "magicoder:7b-s-cl",

    "google/gemma-2-2b-it": "gemma2:27b",
}
if __name__ == "__main__":
    import ollama
    for old, new in legacy_to_ollama.items():
        print("Pulling", new)
        ollama.pull(new)

api_types = {
    # OpenAI
    "gpt3.5": "openai",
    "gpt4": "openai",
    "gpt3.5-turbo": "openai",
    "gpt4-turbo": "openai",

    # Google
    "gemini": "google-beta",
    "palm2": "google",

    # Meta
    "gemma2": "llama",
    "llama2": "llama",
    "llama3": "llama",
    "llama2-7b": "llama",
    "llama2-70b": "llama",
    "llama2-noinstruct": "llama",
    "llama2-7b-noinstruct": "llama",
    "llama2-70b-noinstruct": "llama",
    "codellama": "llama",
    "codellama7": "llama",
    "codellama13": "llama",
    "codellama-noinstruct": "llama",
    "codellama7-noinstruct": "llama",
    "codellama13-noinstruct": "llama",

    "deepseek33": "llama",
    "deepseek6.7": "llama",

    # StarCoder
    "starcoder-ta": "starcoder-ta",
    "starchat-beta": "starchat",
    "starcoder2-15b": "starchat",
    "semcoder": "semcoder",
    "semcoder-s": "semcoder",

    # Mistral
    "mistral": "mistral",
    "mixtral": "mistral",

    # MagiCoder
    "magicoder-cl": "magicoder-cl",
    "magicoder-s-cl": "magicoder-cl",
    "magicoder-ds": "magicoder-ds",
    "magicoder-s-ds": "magicoder-ds",
    "magicoder": "magicoder-cl",

    # WizardCoder
    "wizardcoder": "wizardcoder",
}

# Deprecated models
deprecated_models_ids = {
    # These models don't follow instructions well
    "instructcodet5p": "Salesforce/instructcodet5p-16b",
    "phi-1": "microsoft/phi-1",
    "phi-1.5": "microsoft/phi-1_5",
    "phi-2": "microsoft/phi-2",

    # Ambiguous names
    "llama": "codellama/CodeLlama-34b-Instruct-hf",
    "llama7": "codellama/CodeLlama-7b-Instruct-hf",
    "llama70": "meta-llama/Llama-2-70b-chat-hf",
    "starchat": "HuggingFaceH4/starchat-beta",
    "starcoder": "HuggingFaceH4/starchat-beta",
}
deprecated_api_types = {
    # These models don't follow instructions well
    "instructcodet5p": "instructcodet5",
    "phi-1": "phi",
    "phi-1.5": "phi",
    "phi-2": "phi",

    # Ambiguous names
    "llama": "llama",
    "llama7": "llama",
    "llama70": "llama",
    "starcoder": "starchat",
    "starchat": "starchat",
}
models_ids.update(deprecated_models_ids)
api_types.update(deprecated_api_types)
