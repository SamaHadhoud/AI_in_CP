

import weave
from models import get_vllm, get_embedding_model
from vllm import LLM, SamplingParams
@weave.op
def count_tokens(text: str) -> int:
    return len(get_vllm().tokenizer.encode(text))

@weave.op
def call_model(messages, **kwargs):
    # Preprocess messages to ensure they are in the correct format
    processed_messages = []
    for message in messages:
        if isinstance(message, dict):
            content = message['content']
            role = message['role']
        elif isinstance(message, str):
            # Assume it's a user message if it's a string
            content = message
            role = "user"
        else:
            raise ValueError(f"Unexpected message format: {type(message)}")

        if isinstance(content, list):
            # Join text items and ignore image items
            text_content = ' '.join(item['text'] for item in content if item.get('type') == 'text')
            processed_messages.append({"role": role, "content": text_content})
        else:
            processed_messages.append({"role": role, "content": content})

    # # Format messages for VLLM
    # prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in processed_messages])
    # prompt += "\nAssistant:"

    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4000)

    # Generate
    vllm_instance = get_vllm()
    outputs = vllm_instance.generate(processed_messages, sampling_params)
    
    # Extract and return the generated text
    return outputs[0].outputs[0].text