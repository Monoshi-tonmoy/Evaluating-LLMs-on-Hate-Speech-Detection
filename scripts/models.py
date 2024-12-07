import json
import os
import gc

import backoff
import dotenv
import torch
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import tiktoken


dotenv.load_dotenv()





N_TOKENS = 512
# N_TOKENS = 100


class MyOpenAITokenizer:
    def __init__(self, model_name):
        self.encoding = tiktoken.encoding_for_model(model_name)
    
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.encoding.encode(text)
    
    def decode(self, tokens):
        return self.encoding.decode(tokens)




class Model:
    def __init__(self, model_id, api_type="hf", print_only_mode=False) -> None:
        self.print_only_mode = print_only_mode
        self.model_id = model_id
        self.api_type = api_type
        self.tokenizer = self.initialize_tokenizer()
        self.prefix = None
        self.model_initialized = False
        self.device = None
        self.max_length = self.get_max_length()
    
    def ensure_model_initialized(self):
        if not self.model_initialized:
            self.initialize_model()
    
    def get_max_length(self):
        if self.api_type == "openai":
            return 4097 - 100 - N_TOKENS
        else:
            return None

    def initialize_tokenizer(self):
        if self.api_type == "openai":
            tokenizer = MyOpenAITokenizer(self.model_id)
        elif self.api_type == "google" or self.api_type == "google-beta":
            GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
            genai.configure(api_key=GOOGLE_API_KEY)
            tokenizer = MyOpenAITokenizer("gpt-3.5-turbo") # TODO: use Google's service, currently the API is fucked
        else:
            dotenv.load_dotenv()
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def initialize_model_hf(self):
        """
            This function is to initialize the hf models
        """
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        if self.model_id == "Salesforce/instructcodet5p-16b":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            )
    
    def initialize_model(self):
        """
            This function is to initialize the model
        """
        if self.api_type == "openai":
            dotenv.load_dotenv()
            self.model_ = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif self.api_type == "google-beta":
            dotenv.load_dotenv()
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.model_ = genai.GenerativeModel(self.model_id)
        elif self.api_type == "google":
            dotenv.load_dotenv()
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        else:
            self.initialize_model_hf()
        self.model_initialized = True
        self.device = "cuda"

    def tokenize_llama(self, dialogs, prefix=None):
        inputs = self.tokenizer.encode(
            dialogs,
            add_special_tokens=False
        )
        
        # Create the dialog tokens list
        dialog_tokens = inputs
        
        # Convert to tensor and move to the appropriate device
        return torch.tensor([dialog_tokens]).to(self.device)



    def tokenize_mistral(self, dialogs, prefix):
        dialog_tokens = self.tokenizer.apply_chat_template(dialogs, return_tensors="pt").to(self.device)
        return dialog_tokens




    def tokenize_starchat(self, dialogs, prefix):
        return self.tokenizer.encode(dialogs, return_tensors="pt").to(self.device)

    def tokenize_wizardcoder(self, dialogs, prefix):
        return self.tokenizer.encode(dialogs, return_tensors="pt").to(self.device)



    
    def tokenize_phi(self, dialogs, prefix):
        return self.tokenizer.encode(dialogs, return_tensors="pt").to(self.device)

    
    def tokenize_instructcodet5(self, dialogs, prefix):
        return self.tokenizer.encode(dialogs, return_tensors="pt").to(self.device)


    def tokenize_magicoder(self, dialogs, prefix):
        return self.tokenizer.encode(dialogs, return_tensors="pt").to(self.device)
    
    def tokenize_semcoder(self, dialogs, prefix):
        return self.tokenizer.encode(dialogs, return_tensors="pt").to(self.device)

    def tokenize(self, dialogs):
        """
            This models is to tokenize the dialogs.
        """
        if self.api_type == "llama":
            return self.tokenize_llama(dialogs)
        elif self.api_type == "starchat":
            return self.tokenize_starchat(dialogs)
        elif self.api_type == "starcoder-ta":
            return self.tokenize_starcoder_techassistant(dialogs)
        elif self.api_type == "wizardcoder":
            return self.tokenize_wizardcoder(dialogs)
        elif self.api_type== "mistral":
            if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
                return self.tokenize_mixtral(dialogs)
            else:
                return self.tokenize_mistral(dialogs)
        elif self.api_type=="instructcodet5":
            return self.tokenize_instructcodet5(dialogs)
        elif self.api_type=="phi":
            return self.tokenize_phi(dialogs)
        elif self.api_type in ("magicoder-cl", "magicoder-ds"):
            return self.tokenize_magicoder(dialogs)
        elif self.api_type == "openai":
            message_tokens = [self.tokenizer.encode(d["content"]+"\n") for d in dialogs]
            all_tokens = []
            for mt in message_tokens:
                all_tokens.extend(mt)
            return [all_tokens]
        elif self.api_type == "semcoder":
            return self.tokenize_semcoder(dialogs)
        # elif self.api_type == "gemini":
        #     raise NotImplementedError
        else:
            raise NotImplementedError(self.api_type)


    @backoff.on_exception(backoff.expo, Exception, max_tries=8)
    def query_google_beta(self, dialogs, return_metadata):
        """
        Query the Gemini Google Chat API.
        """

        if dialogs[0]["role"] == "system":
            system_message = dialogs.pop(0)["content"] + " "
        else:
            system_message = ""
        
        history, query = dialogs[:-1], dialogs[-1]["content"]
        query = system_message + query
        history = [{"parts": {"text": d["content"]}, **{k: v for k, v in d.items() if k != "content"}} for d in history]
        for h in history:
            if h["role"] == "assistant":
                h["role"] = "model"

        chat = self.model_.start_chat(history=history,
            # context=system_message,
            )
        completion = chat.send_message(query,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=N_TOKENS,
                temperature=0.1,
            ),
        )
        try:
            text = completion.text
        except Exception:
            text = None
            print("Error getting text:", query)

        if return_metadata:
            return {
                "input_json": json.dumps(dialogs),
                "output_completion": glm.Candidate.to_dict(completion.candidates[0]),
                "output_tokens": [],
                "output": text,
                "output_without_input": text,
                "query": query,
                "history": history,
            }
        else:
            return text


    def query_google(self, dialogs, return_metadata):
        """
        Query the Google Chat API.
        """

        if dialogs[0]["role"] == "system":
            system_message = dialogs[0]["content"]
            messages = dialogs[1:]
        else:
            system_message = None
            messages = dialogs
        messages = [d["content"] for d in messages]
        completion = genai.chat(
            model=f"models/{self.model_id}",
            context=system_message, messages=messages,
            temperature=0.1, candidate_count=1
        )

        if return_metadata:
            return {
                "input_json": json.dumps(dialogs),
                "output_completion": glm.Candidate.to_dict(completion.candidates[0]),
                "output_tokens": [],
                "output": completion.text,
                "output_without_input": completion.text,
            }
        else:
            return completion.text


    @backoff.on_exception(backoff.expo, openai.RateLimitError, giveup=lambda ex: "You exceeded your current quota" in str(ex))
    def query_openai(self, dialogs, return_metadata):

        completion = self.model_.chat.completions.create(
            model=self.model_id, 
            messages=dialogs,
            n=1,
            temperature=0.1,
            max_tokens=N_TOKENS,
        )
        content = completion.choices[0].message.content
        if return_metadata:
            return {
                "input_json": json.dumps(dialogs),
                "output_completion": completion.model_dump(),
                "output_tokens": [],
                "output": content,
                "output_without_input": content,
            }
        else:
            return content

    def query_hf(self, dialogs, return_metadata):
        input_ids = self.tokenize(dialogs)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=N_TOKENS,
                # do_sample=False,
                num_return_sequences=5,
                do_sample=True,
                top_p=0.9,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        #output = output.to("cpu")[0]
        output = output[0].to("cpu")
        if return_metadata:
            ret = {
                "input_json": json.dumps(dialogs),
                "output_tokens": self.tokenizer.convert_ids_to_tokens(output),
                "output": self.tokenizer.decode(output),
                "output_without_input": self.tokenizer.decode(output[input_ids.shape[1]:]),
            }
        else:
            ret = self.tokenizer.decode(output)
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        return ret

    def query(self, dialogs, return_metadata=False):
        if self.api_type == "openai":
            return self.query_openai(dialogs, return_metadata=return_metadata)
        elif self.api_type == "google-beta":
            return self.query_google_beta(dialogs, return_metadata=return_metadata)
        elif self.api_type == "google":
            return self.query_google(dialogs, return_metadata=return_metadata)
        else:
            return self.query_hf(dialogs, return_metadata=return_metadata)
