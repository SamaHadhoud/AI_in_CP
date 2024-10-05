from dataclasses import dataclass
from pathlib import Path
import logging

import weave
import simple_parsing
from vllm import LLM, SamplingParams
from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run,TimeoutException
import re
from transformers import AutoTokenizer
import torch

import os

import asyncio
import json
import logging
from pathlib import Path
import time
from typing import Optional, List
from rich.logging import RichHandler

import os
from vllm import LLM, SamplingParams
from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run, TimeoutException
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
from dataclasses import dataclass, field

llm = LLM(model=model_name, dtype="float16")

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
    outputs = llm.generate([prompt], sampling_params)
    # Return the generated text
    return outputs[0].outputs[0].text.strip()

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

    # Let's make a second call to the model to extract the code from the response
    messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": [
        {"type": "text", 
         "text": extract_prompt}
    ]})

    # call model second time to extract the code
    solution = call_model(messages=messages)
    logging.info("Extracting the solution from the previous generation...")

    code_match = re.search(r'```python\n(.*?)```', solution, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
    else:
        extracted_code = ""
        logging.error("No Python code found in the solution")


    # in case we have ```python stuff...`
    # solution = maybe_remove_backticks(solution)
    return extracted_code

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
Problem Description:
{problem_description}

Sample Input:
{sample_input}

Sample Output:
{sample_output}

Your task is to create a Python function named 'solve' that accurately and efficiently solves the problem described above. The function should take a string input similar to the sample input and return a string output similar to the sample output.

Requirements:
1. Implement the 'solve' function with the signature: def solve(input_data: str) -> str:
2. Parse the input string correctly and handle all aspects of the problem.
3. Optimize the algorithm for both time and space complexity.
4. Use 'tqdm' for progress bars in loops processing large datasets, if applicable.
5. Include necessary imports at the beginning of your code.
6. Provide brief comments explaining any complex logic or optimizations.

The file should have a single `solve` method with the following signature:

```python
from tqdm import tqdm

def solve(input_data: str) -> str:
    # Your implementation here
"""

extract_prompt = """
Extract the code from the response. reply with the code only. Omit any additional example or explanation.
- If the solution involves a for loop, please use `for sample in tqdm(range(samples))` to show progress.
- The code should be a valid python program.
- Get the `solve` function with the corresponding imports"""


@dataclass
class SolutionAttempt:
    code: str
    status: str
    test_cases: dict = None
    error: str = None
    execution_time: float = None

@weave.op
def solve_problem(problem: Problem, use_images=False, timeout=60) -> SolutionAttempt:
    code = generate_code(
        problem, 
        system_prompt=system_prompt, 
        prompt_template=prompt_template, 
        extract_prompt=extract_prompt, 
        use_images=use_images)

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

def rank_solutions(solutions: List[SolutionAttempt]) -> List[SolutionAttempt]:
    def solution_score(solution: SolutionAttempt) -> tuple:
        if solution.status == "success":
            return (2, solution.test_cases['len_passed_cases'], -solution.execution_time)
        elif solution.status == "timeout":
            return (1, 0, 0)
        else:  # runtime_error
            return (0, 0, 0)
    
    return sorted(solutions, key=solution_score, reverse=True)

@weave.op
async def reflection(problem: Problem, solution_result: SolutionAttempt) -> str:
    error_str = f"Error: {solution_result.error}" if solution_result.error else ""
    test_cases_str = json.dumps(solution_result.test_cases, indent=2) if solution_result.test_cases else ""
    Offending = ""
    if(test_cases_str):
        Offending = f"Code Current output: {solution_result.test_cases['actual']}\nOffending Test Cases:\n"
        for case in solution_result.test_cases["offending_cases"]:
            Offending+=f"line {case[0]}, should be {case[1]} instead of {case[2]}\n"
    reflection_prompt = f"""
You are a world-class competitive programmer with a keen eye for detail and problem solving. 
Your expertise is in algorithms and data structures. 
You have incorrectly answered the following programming problem. 
Your task is to reflect on the problem, your solution, and the correct answer.
You will then use this information help you answer the same question in the future. 
First, explain why you answered the question incorrectly.
Secondly, create a list of detailed instructions to help you correctly solve this problem in the future.
Be concise in your response; however, capture all of the essential information.

Problem:
{problem.problem_description}

Sample Input:
{problem.sample_input}

Sample Output:
{problem.sample_output}

<incorrect_solution>
{solution_result.code}
</incorrect_solution>
<test_report>
{"Status: " + solution_result.status if solution_result.status != "success" else ""}
{error_str}
{Offending if Offending else ""}
</test_report>

**Format Instructions: Your response must follow the following xml format** -

<root>
<reflection>
[Reflect on the problem, your solution, and the correct answer.]
</reflection>
<instructions>
[Create a list of detailed instructions to help you correctly solve this problem in the future.]
</instructions>
</root>
---
Let's think step by step to reflect on the problem:
"""

    messages = [
        {"role": "system", "content": reflection_prompt}
        # ,
        # {"role": "user", "content": reflection_prompt},
    ]

    reflection_response = call_model(messages=messages)
    return reflection_response

@weave.op
async def improve_solution(problem: Problem, previous_solution: SolutionAttempt, reflection: str = "") -> str:
    error_str = f"Error: {previous_solution.error}" if previous_solution.error else ""
    test_cases_str = json.dumps(previous_solution.test_cases, indent=2) if previous_solution.test_cases else ""
    Offending = ""
    if(test_cases_str):

        Offending = f"Code Current output: {previous_solution.test_cases['actual']}\nOffending Test Cases:\n"
        for case in previous_solution.test_cases["offending_cases"]:
            Offending+=f"line {case[0]}, should be {case[1]} instead of {case[2]}\n"
    improve_prompt = f"""
You have incorrectly answered the following programming problem. Based on the following reflection and improvements, please provide an improved solution to the problem:

Problem:
{problem.problem_description}

Sample Input:
{problem.sample_input}

Sample Output:
{problem.sample_output}

<incorrect_solution>
{previous_solution.code}
</incorrect_solution>
<test_report>
{"Status: " + previous_solution.status if previous_solution.status != "success" else ""}
{error_str}
{Offending if Offending else ""}
</test_report>

Reflection and improvements:
{reflection}

Please provide an improved solution that addresses the issues identified in the reflection.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": improve_prompt},
    ]

    improved_solution = call_model(messages=messages)

    
    logging.info("Generating initial analysis and solution")


    # Let's make a second call to the model to extract the code from the response
    messages.append({"role": "assistant", "content": improved_solution})
    messages.append({"role": "user", "content": [
        {"type": "text", 
         "text": extract_prompt}
    ]})

    # call model second time to extract the code
    solution = call_model(messages=messages)
    logging.info("Extracting the solution from the previous generation...")

    code_match = re.search(r'```python\n(.*?)```', solution, re.DOTALL)
    if code_match:
        extracted_code = code_match.group(1).strip()
    else:
        extracted_code = ""
        logging.error("No Python code found in the solution")


    # in case we have ```python stuff...`
    # solution = maybe_remove_backticks(solution)
    return extracted_code

@dataclass
class Args(simple_parsing.Serializable):
    folder_path: Path = Path("./data/")
    problem_names: List[str] = field(default_factory=list)
    weave_log: bool = False
    use_images: bool = False
    save_output: bool = True
    debug: bool = False
    timeout: int = 60
    max_attempts: int = 6

    def __post_init__(self):
        if not self.problem_names:
            self.problem_names = self.get_problem_names_from_folders()


    def get_problem_names_from_folders(self) -> List[str]:
        problem_folders = []
        for entry in os.scandir(self.folder_path):
            if entry.is_dir():
                problem_folders.append(entry)
        
        # Sort folders by creation time
        problem_folders.sort(key=lambda x: x.stat().st_ctime)
        
        return [folder.name for folder in problem_folders]
    
@weave.op()
async def solve_single_problem(args: Args, problem_name: str):
    """Solve a single problem and log results in Weave."""
    problem = Problem.from_name(problem_name, args.folder_path)
    logging.info(f"Solving problem: {problem_name}")
    
    solution_attempts = []
    initial_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout)
    solution_attempts.append(initial_solution)

    weave.save({f"{problem_name}_initial_attempt": initial_solution})
    logging.info(f"Initial attempt - Status: {initial_solution.status}")
    

    if not (initial_solution.status == 'success' and initial_solution.test_cases['matches']):
        current_solution = initial_solution
        for attempt in range(args.max_attempts):
            
            logging.info(f"Attempt {attempt + 1} - Reflecting and improving...")
            reflection_result = await reflection(problem, current_solution)
            improved_solution = await improve_solution(problem, current_solution,reflection_result )
            
            solution_result = try_solution(problem, improved_solution, args.timeout)
            solution_attempts.append(solution_result)
            current_solution = solution_result

            logging.info(f"Attempt {attempt + 1} - Status: {solution_result.status}")
            if(solution_result.test_cases):
                logging.info(f"Attempt {attempt + 1} - test_cases: {solution_result.test_cases}")

            if solution_result.status == 'success' and solution_result.test_cases['matches']:
                break

    ranked_solutions = rank_solutions(solution_attempts)
    best_solution = ranked_solutions[0]
    
    logging.info(f"Best solution status: {best_solution.status}")

    # Check full input on best solution
    logging.info("> Checking full input on best solution...")
    best_full_input_result = solve_full_input(problem, best_solution, args)

    return {
        "initial_solution": initial_solution,
        "best_solution": best_solution,
    }

@weave.op()
def try_solution(problem: Problem, code: str, timeout: int) -> SolutionAttempt:
    """Try a solution and return a SolutionAttempt object."""
    input_data, output = problem.sample_input, problem.sample_output
    
    try:
        start_time = time.time()
        generated_output = run(code, input=input_data, timeout=timeout)
        execution_time = time.time() - start_time
        test_cases = check_solution(output, generated_output)
        return SolutionAttempt(code=code, status="success", test_cases=test_cases, execution_time=execution_time)
    except TimeoutException:
        return SolutionAttempt(code=code, status="timeout", error="Execution time limit exceeded")
    except Exception as e:
        return SolutionAttempt(code=code, status="runtime_error", error=str(e))

@weave.op()
def solve_full_input(problem: Problem, solution: SolutionAttempt, args: Args) -> dict:
    """Solve the problem with full input and return the results."""
    result = {}
    
    try:
        start_time = time.time()
        generated_output = run(solution.code, input=problem.get_input(), timeout=args.timeout)
        execution_time = time.time() - start_time
        

        problem.save_output(generated_output)
        problem.save_code(solution.code)
        result["output_saved"] = True
    except TimeoutException:
        logging.error("The solution took too long to execute on the full input and was terminated.")
        logging.error(f"Trying with {2*args.timeout} seconds...")
        try:
            start_time = time.time()
            generated_output = run(solution.code, input=problem.get_input(), timeout=2*args.timeout)
            execution_time = time.time() - start_time
            
            
            problem.save_output(generated_output)
            problem.save_code(solution.code)
            result["output_saved"] = True
        except TimeoutException:
            result = {
                "status": "timeout",
                "error": "The solution took too long to execute even with extended timeout."
            }
            problem.save_code(solution.code)
    except Exception as e:
        print(e)
        result = {"status": "error", "error": str(e)}
        problem.save_code(solution.code)
    
    return result

@weave.op()
async def main(args: Args):
    """Main function to solve multiple problems sequentially."""
    setup_logger(args.debug)

    if args.weave_log:
        weave.init("hack-starter")

    all_results = {}
    total_start_time = time.time()
    for problem_name in args.problem_names:
        problem_results = await solve_single_problem(args, problem_name)
        all_results[problem_name] = problem_results
        
        # logging.info(f"Results for {problem_name}:")
        logging.info(f"Initial solution status: {problem_results['initial_solution'].status}")
        # logging.info(f"Initial full input result: {problem_results['initial_full_input_result']}")
        logging.info(f"Best solution status: {problem_results['best_solution'].status}")
        # logging.info(f"Best full input result: {problem_results['best_full_input_result']}")
        # logging.info("---")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for all problems: {total_duration:.2f} seconds")

if __name__ == "__main__":
    args = simple_parsing.parse(Args)
    asyncio.run(main(args))