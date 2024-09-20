from dataclasses import dataclass
from pathlib import Path
import logging

import openai
import weave
import simple_parsing

from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, maybe_remove_backticks_cpp, check_solution, setup_logger, run_cpp

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
# client = openai.OpenAI()

# @weave.op
# def call_model(messages, **kwargs):
#     return client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         **kwargs
#     ).choices[0].message.content


model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()

# Set a distinct pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

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

    # Apply chat template
    inputs = tokenizer.apply_chat_template(processed_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    attention_mask = inputs.ne(tokenizer.pad_token_id).int()
    
    # Move both inputs and attention_mask to the model's device
    inputs = inputs.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Generate
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,  # Pass the attention mask
        max_new_tokens=512, 
        do_sample=False, 
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad_token_id
        **kwargs
    )


    # Decode and return the response
    result = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return result.strip()

@weave.op
def generate_code(
    problem: Problem, 
    system_prompt: str, 
    prompt_template: str, 
    extract_prompt: str,
    use_images: bool = False) -> str:
    logging.info(f"Generating code solution for: {problem.name}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output,
            )}
        ] + ([{"type": "image_url", "image_url": {"url": img}} for img in problem.images] if use_images else [])}
    ]
    
    # call model one first time to get the code
    out = call_model(messages=messages)
    logging.info("Generating initial analysis and solution")

#     # Additional prompt to ensure correct input handling
    
#     messages.append({"role": "assistant", "content": out})
#     messages.append({"role": "user", "content": [
#         {"type": "text", 
#         "text": f"""
# Please ensure the solve function correctly handles the Input and Output format with multiple test cases. 

# Remember, Input Format:

# Input begins with an integer \\(T\\), the number of test cases. Each case contains one line with three space-separated integers, \\(A\\), \\(B\\) and \\(C\\).


# Output Format:

# For the \\(i\\)th test case, print "`Case #i:` " followed by the largest possible \\(K\\) for which you can build a \\(K\\)-decker cheeseburger, or \\(0\\) if you cannot build even a \\(1\\)-decker cheeseburger.

# Sample Input:
# {problem.sample_input}

# Sample Output:
# {problem.sample_output}

# Please modify the solve function to correctly handle this input format and produce the expected output.
# """}
# ]})


#     # Generate final code
#     solution = call_model(messages=messages)

    # Let's make a second call to the model to extract the code from the response
    messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": [
        {"type": "text", 
         "text": extract_prompt}
    ]})

    # call model second time to extract the code
    solution = call_model(messages=messages)
    print(solution)
    logging.info("Extracting the solution from the previous generation...")

    # in case we have ```cpp stuff...`
    solution = maybe_remove_backticks_cpp(solution)
    return solution

system_prompt = "You are an expert problem solver. Your task is creating the code to solve the problem at hand in cpp."

prompt_template = """
Problem: 
{problem_description}

Input: 
{sample_input}

Output: 
{sample_output}

Create a standalone C++ program that solves this problem. Follow these guidelines:

1. Include all necessary headers.
2. Implement a 'solve' function that processes the input and returns the solution.
3. Implement a main() function that reads input from stdin and writes output to stdout.
4. Handle multiple test cases as specified in the problem.

Here's the structure your code should follow:


#include <iostream>
#include <string>
#include <vector>
#include <sstream>
// Add any other necessary includes

std::string solve(const std::string& input) {{
    // Your solution here
    // Process input and return the result as a string
}}
"""

extract_prompt = """
Extract the C++ code from the response. Reply with the complete, standalone C++ code only. Omit any additional explanation or comments.
- The code should be a valid C++ program that can be compiled and run independently.
- Include all necessary headers, the 'solve' function, and the 'main' function.
- Ensure the program reads input from stdin and writes output to stdout.
- The code should handle multiple test cases as specified in the problem."""

@weave.op
def solve_problem(problem: Problem, use_images=False, timeout=60) -> dict:
    code = generate_code(
        problem, 
        system_prompt=system_prompt, 
        prompt_template=prompt_template, 
        extract_prompt=extract_prompt, 
        use_images=use_images)
    print(code)
    input_data, output = problem.sample_input, problem.sample_output
    input_bytes = input_data.encode() if isinstance(input_data, str) else input_data
    generated_output = run_cpp(code, input=input_bytes, timeout=timeout) 
    
    return {"code": code, "generated_output": generated_output, "expected_output": output}

@dataclass
class Args(simple_parsing.Serializable):
    problem_name: str = "cheeseburger_corollary_ch1" # name of the problem to solve
    folder_path: Path = Path("./dataset/2023/practice/") # path to the folder containing the problems
    weave_log: bool = False # set to True to log to weave
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
    problem_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout)
    matches = check_solution(problem_solution["expected_output"], problem_solution["generated_output"])
    logging.info("Sample Matches:")
    logging.info(matches)

    logging.info("> Solving on full input...")
    expected_output = problem.get_output()
    input_bytes = problem.get_input().encode() if isinstance(problem.get_input(), str) else problem.get_input()
    generated_output = run_cpp(problem_solution["code"], input=input_bytes, timeout=args.timeout) 
    matches = check_solution(expected_output, generated_output)
    logging.info("Final Matches:")
    logging.info(matches)

    if args.save_output:
        logging.info("> Saving output to files")
        problem.save_output(problem_solution["generated_output"])
        problem.save_code(problem_solution["code"])

