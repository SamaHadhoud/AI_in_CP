from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from huggingface_hub import login
from pathlib import Path

def get_hugging_face_token(token_file: str = "../TOKEN") -> str:
    """
    Reads the Hugging Face token from an external file.

    Args:
        token_file (str): Path to the file containing the token.

    Returns:
        str: The Hugging Face token.
    """
    token_path = Path(token_file)
    if not token_path.exists():
        raise FileNotFoundError(f"Token file not found at: {token_file}")
    
    with open(token_path, "r") as file:
        token = file.read().strip()
    return token

# Retrieve token and login
try:
    token = get_hugging_face_token()
    login(token=token)
    print("Logged into Hugging Face successfully.")
except Exception as e:
    print(f"Failed to log in: {e}")


# Add Hugging Face login
login(token=token)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

class VLLMSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("Initializing vLLM model for the first time...")
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        self.llm = LLM(model=model_name, 
                       trust_remote_code=True, 
                       enforce_eager=True,
                       dtype="float16", 
                       gpu_memory_utilization=0.9,
                       max_model_len=12000)
        
        self.default_sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=8000)
        
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompts, sampling_params=None):

        text = self.tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True
        )
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        outputs = self.llm.generate([text], sampling_params)
        return outputs

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

def get_vllm():
    return VLLMSingleton.get_instance()