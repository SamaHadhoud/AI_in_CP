from dataclasses import dataclass
from pathlib import Path
import logging
from dataclasses import dataclass, field
import weave
import simple_parsing
from vllm import LLM, SamplingParams
from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run,TimeoutException
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from models import get_vllm, get_embedding_model
import time
from bla import *
import asyncio
vllm = get_vllm()

@weave.op
def count_tokens(text: str) -> int:
    return len(vllm.tokenizer.encode(text))

@dataclass
class SolutionAttempt:
    code: str
    status: str
    test_cases: dict = None
    error: str = None
    execution_time: float = None
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

system_prompt="""
You are an expert Python developer specializing in algorithmic problem-solving. Your task is to create highly efficient and accurate Python code that precisely addresses the given problem. Your solution should prioritize correctness, optimal performance, and adherence to all problem specifications.

Key Requirements:
1. Use the function signature: 'def solve(input_data: str) -> str:'
2. Implement an algorithm that correctly solves all aspects of the problem, including edge cases.
3. Optimize for both time and space complexity where possible, without compromising correctness.
4. Use 'from tqdm import tqdm' for progress bars in loops processing large datasets.
5. Include all necessary imports at the beginning of your code.
6. Handle input and output as strings, parsing and formatting as required.
7. Provide clear, concise comments explaining complex logic or optimizations.

Best Practices:
- Carefully analyze the problem description to identify all requirements and constraints.
- Consider various algorithmic approaches and choose the most efficient one for the given problem.
- Implement robust error handling and input validation where appropriate.
- Use appropriate data structures to optimize time and space complexity.
- Write clean, readable code following PEP 8 style guidelines.
- If applicable, consider using Python's built-in functions and libraries for optimization.

Remember: Your primary goal is to create a solution that is both correct and efficient, capable of handling all possible inputs within the problem's constraints.
"""

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
    try:
        start_time = time.time()
        generated_output = run(code, input=input_data, timeout=timeout)
        execution_time = time.time() - start_time
        test_cases = check_solution(output, generated_output)
        return SolutionAttempt(code=code, status="success", test_cases=test_cases, execution_time=execution_time)
    except TimeoutException:
        return SolutionAttempt(code=code, status="timeout", error= "Execution time limit exceeded")
    except Exception as e:
        return SolutionAttempt(code=code, status="runtime_error", error=str(e))


    
