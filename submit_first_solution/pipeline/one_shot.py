from dataclasses import dataclass
from pathlib import Path
import logging

import weave
import simple_parsing
from vllm import LLM, SamplingParams
from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run,TimeoutException
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
# model_name = "NTQAI/Nxcode-CQ-7B-orpo"
# model_name = "codellama/CodeLlama-7b-Instruct-hf"
# Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
from models import get_vllm, get_embedding_model

vllm = get_vllm()

@weave.op
def count_tokens(text: str) -> int:
    return len(vllm.tokenizer.encode(text))



# llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.95)

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

    # Format messages for VLLM
    prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in processed_messages])
    prompt += "\nAssistant:"

    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4000)

    # Generate
    vllm_instance = get_vllm()
    outputs = vllm_instance.generate([prompt], sampling_params)
    
    # Extract and return the generated text
    return outputs[0].outputs[0].text

# @weave.op
# def call_model(messages, **kwargs):
#     # Preprocess messages to ensure they are in the correct format
#     processed_messages = []
#     for message in messages:
#         if isinstance(message, dict):
#             content = message['content']
#             role = message['role']
#         elif isinstance(message, str):
#             # Assume it's a user message if it's a string
#             content = message
#             role = "user"
#         else:
#             raise ValueError(f"Unexpected message format: {type(message)}")

#         if isinstance(content, list):
#             # Join text items and ignore image items
#             text_content = ' '.join(item['text'] for item in content if item.get('type') == 'text')
#             processed_messages.append({"role": role, "content": text_content})
#         else:
#             processed_messages.append({"role": role, "content": content})

#     # Format messages for VLLM
#     prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in processed_messages])
#     prompt += "\nAssistant:"

#     # Set up sampling parameters
#     sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4000)


#     # Generate
#     outputs = vllm.generate([prompt], sampling_params)
#     print(prompt)
#     prompt_tokens = count_tokens(prompt)
#     print(f"Prompt Tokens: {prompt_tokens}")
#     outputs_tokens = count_tokens(outputs[0].outputs[0].text.strip())
#     print(f"outputs Tokens: {outputs_tokens}")
#     # Return the generated text
#     return outputs[0].outputs[0].text.strip()

@weave.op
def generate_code(
    problem: Problem, 
    system_prompt: str, 
    prompt_template: str, 
    extract_prompt: str,
    use_images: bool = False,
    max_attempts: int = 3, 
    examples:str="") -> str:
    logging.info(f"Generating code solution for: {problem.name}")

    for attempt in range(max_attempts):
        logging.info(f"Generating code solution for: {problem.name} (Attempt {attempt + 1})")

    # Count tokens for problem components
    problem_description_tokens = count_tokens(problem.problem_description)
    sample_input_tokens = count_tokens(problem.sample_input)
    sample_output_tokens = count_tokens(problem.sample_output)
    total_problem_tokens = problem_description_tokens + sample_input_tokens + sample_output_tokens

    # Count tokens for prompts
    system_prompt_tokens = count_tokens(system_prompt)
    
    # Format the prompt template with problem details
    formatted_prompt = prompt_template.format(
        problem_description=problem.problem_description,
        sample_input=problem.sample_input,
        sample_output=problem.sample_output,
        examples=examples
    )
    prompt_template_tokens = count_tokens(formatted_prompt)
    print(formatted_prompt)
    
    extract_prompt_tokens = count_tokens(extract_prompt)

    # Print token counts
    print(f"Token counts:")
    print(f"  Problem description: {problem_description_tokens}")
    print(f"  Sample input: {sample_input_tokens}")
    print(f"  Sample output: {sample_output_tokens}")
    print(f"  Total problem: {total_problem_tokens}")
    print(f"  System prompt: {system_prompt_tokens}")
    print(f"  Prompt template: {prompt_template_tokens}")
    print(f"  Extract prompt: {extract_prompt_tokens}")
    print(f"  Total prompts: {system_prompt_tokens + prompt_template_tokens + extract_prompt_tokens}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
                examples=examples
            )}
        ] + ([{"type": "image_url", "image_url": {"url": img}} for img in problem.images] if use_images else [])}
    ]

    # call model one first time to get the code
    out = call_model(messages=messages)

    logging.info("Generating initial analysis and solution")


    # Let's make a second call to the model to extract the code from the response
    messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": [
        {"type": "text", 
         "text": extract_prompt}
    ]})

    # call model second time to extract the code
    solution = call_model(messages=messages)
    print("******************************************************")
    print(solution)
    logging.info("Extracting the solution from the previous generation...")

    code_match = re.search(r'```python\n(.*?)```', solution, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
            
        if extracted_code:
            return extracted_code
        else:
            logging.error("Extracted code is empty")
    else:
        logging.error("No Python code found in the solution")
        
    if attempt < max_attempts - 1:
        logging.warning(f"Attempt {attempt + 1} failed to produce valid code. Retrying...")
    
    logging.error(f"Failed to generate valid code after {max_attempts} attempts")
    return "# Failed to generate valid code"

system_prompt = " You are an expert problem solver. Your task is creating the code to solve the problem at hand in python."

prompt_template = """
Problem: 
{problem_description}

Input: 
{sample_input}

Output: 
{sample_output}

You have previously solved similar problems like the following:
Examples:
{examples}


I want you to get inspired from them and create a Python program that solves the current problem. Your solution must include a function named 'solve' with the following signature:

def solve(input_data: str) -> str:
    # Your code here

The 'solve' function should take the input as a string and return the output as a string.

Please provide only the Python code, enclosed in triple backticks, like this:

```python
# Your imports here
from tqdm import tqdm

def solve(input_data: str) -> str:
    # Your code here
"""


extract_prompt = """
Extract the complete Python code from the previous response. The code should:
1. Be enclosed in triple backticks with the Python language specifier.
2. Include all necessary imports at the top.
3. Contain a 'solve' function with the signature: def solve(input_data: str) -> str:
4. Use 'for sample in tqdm(range(samples))' for any loops to show progress.
5. Be a complete, runnable Python program.

Provide only the code, without any additional explanations or comments. The response should look like this:

```python
# Imports
from tqdm import tqdm

def solve(input_data: str) -> str:
    # Function implementation
"""
@weave.op
def solve_problem(problem: Problem, use_images=False, timeout=60, examples="") -> dict:
    code = generate_code(
        problem, 
        system_prompt=system_prompt, 
        prompt_template=prompt_template, 
        extract_prompt=extract_prompt, 
        use_images=use_images,
        examples=examples)
    print("**************************************************************")
    print(code)

    input_data, output = problem.sample_input, problem.sample_output
    generated_output = run(code, input=input_data, timeout=timeout) 
    
    return {"code": code, "generated_output": generated_output, "expected_output": output}

@dataclass
class Args(simple_parsing.Serializable):
    problem_name: str = "dim_sum_delivery" # name of the problem to solve
    folder_path: Path = Path("dataset/2023/practice/") # path to the folder containing the problems
    weave_log: bool = True # set to True to log to weave
    use_images: bool = False # set to True to use images in the prompt
    save_output: bool = True # set to True to save the output to a file
    debug: bool = False # set to True to see the debug logs
    timeout: int = 60 # timeout for the code execution

if __name__=="__main__":
    args = simple_parsing.parse(Args)

    setup_logger(args.debug)

    problem = Problem.from_name(args.problem_name, args.folder_path)

    if args.weave_log: 
        weave.init("hack-starter")
    
    logging.info("> Solving on sample input...")
    try:
        problem_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout)
    except TimeoutException:
        print("The solution took too long to execute and was terminated.")
        problem_solution = None  # or some default value
    matches = check_solution(problem_solution["expected_output"], problem_solution["generated_output"])
    logging.info("Sample Matches:")
    logging.info(matches)

    logging.info("> Solving on full input...")
    expected_output = problem.get_output()
    generated_output = run(problem_solution["code"], input=problem.get_input(), timeout=args.timeout) 
    matches = check_solution(expected_output, generated_output)
    logging.info("Final Matches:")
    logging.info(matches)

    if args.save_output:
        logging.info("> Saving output to files")
        problem.save_output(problem_solution["generated_output"])
        problem.save_code(problem_solution["code"])

