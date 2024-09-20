import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
import time

import openai
import weave
import simple_parsing
from tqdm.asyncio import tqdm

from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, setup_logger, check_solution, arun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
# client = openai.AsyncOpenAI()


# @weave.op
# async def call_model(messages, **kwargs):
#     out = await client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         **kwargs
#     )
#     return out.choices[0].message.content



model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()

# Set a distinct pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

@weave.op
async def call_model(messages, **kwargs):
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
async def generate_code(
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
    out = await call_model(messages=messages)
    logging.info("Generating initial analysis and solution")

    # Let's make a second call to the model to extract the code from the response
    messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": [
        {"type": "text", 
         "text": extract_prompt}
    ]})

    # call model second time to extract the code
    solution = await call_model(messages=messages)
    logging.info("Extracting the solution from the previous generation...")

    # in case we have ```python stuff...`
    solution = maybe_remove_backticks(solution)
    return solution

system_prompt = "You are an expert problem solver. Your task is creating the code to solve the problem at hand in python."

prompt_template = """
Problem: 
{problem_description}

Input: 
{sample_input}

Output: 
{sample_output}

Create a python program that returns the correct output for the given input. 
Make the code efficient and fast, so we can solve large inputs.
The file should have a single `solve` method that has the following signature:
input: [str]: The same Input provided above
output [str]: The same Output provided above

```python
from tqdm import tqdm
def solve(input: str) -> str: 
```
"""

extract_prompt = """
Extract the code from the response. reply with the code only. Omit any additional example or explanation.
- If the solution involves a for loop, please use `for sample in tqdm(range(samples))` to show progress.
- The code should be a valid python program.
- Get the `solve` function with the corresponding imports"""

@dataclass
class Args(simple_parsing.Serializable):
    folder_path: Path = Path("./dataset/2023/practice") # path to the folder containing the problems
    weave_log: bool = False # set to True to log to weave
    weave_eval: bool = False # set to True to evaluate the code
    max_num_problems: int = 5 # maximum number of problems to evaluate
    on_sample: bool = True # run evaluation on sample inputs/outputs
    use_images: bool = False # set to True to use images in the prompt
    save_output: bool = True # set to True to save the output to a file
    debug: bool = False # set to True to see the debug logs
    timeout: int = 60 # timeout for the code execution

if __name__=="__main__":
    args = simple_parsing.parse(Args)
    setup_logger(args.debug)
    logging.info(f"Parsed args: {args}")
    t0 = time.perf_counter()

    problems = Problem.find_all(args.folder_path)[:args.max_num_problems] # dataset

    if args.weave_log: weave.init("hack-starter")

    @weave.op
    async def solve_problem(problem: Problem, on_sample=args.on_sample) -> dict:
        code = await generate_code(
            problem, 
            system_prompt=system_prompt, 
            prompt_template=prompt_template, 
            extract_prompt=extract_prompt, 
            use_images=args.use_images)
        if on_sample:
            input, output = problem.sample_input, problem.sample_output
        else:
            input, output = problem.get_input(), problem.get_output()
        generated_output = await arun(code, input=input, timeout=args.timeout) 
        if args.save_output:
            problem.save_output(generated_output)
            problem.save_code(code)
        return {"code": code, "generated_output": generated_output, "expected_output": output}

    def match(model_output: str):
        matches = check_solution(model_output["expected_output"], model_output["generated_output"])
        return matches


    if args.weave_eval:
        dataset = [{"problem": problem} for problem in problems]
        evaluation = weave.Evaluation(dataset=dataset, scorers=[match])
        asyncio.run(evaluation.evaluate(solve_problem))
    else:
        @weave.op
        async def solve_one(problem):
            try:
                model_output = await solve_problem(problem)
                matches = match(model_output)
                logging.info(f"Problem {problem.name} results: {matches}")
                return {"runs": "✅", "error": None, **matches}
            except Exception as e:
                logging.error(f"Problem {problem.name} failed with error: {e}")
                return {"runs": "❌", "error": str(e), "matches": -1, "total": -1, "offending_cases": []}

        @weave.op
        async def evaluate():
            tasks = [solve_one(problem) for problem in problems]
            eval_results = await tqdm.gather(*tasks, desc="Solving problems...")
            return eval_results
        
        eval_results = asyncio.run(evaluate())

        # let's format the results in a pandas dataframe
        import pandas as pd
        from tabulate import tabulate

        df = pd.DataFrame([
            {
                "problem": problem.name,
                "runs": result["runs"],
                "error": result["error"],
                "matches": result["matches"],
                "offending_cases": len(result["offending_cases"]),
                "total": result["total"],
                "valid": "✅" if result["matches"] == result["total"] else "❌"
            }
            for problem, result in zip(problems, eval_results)
        ])
        logging.info("Evaluation results:")
        table = tabulate(df, headers='keys', tablefmt='pretty', showindex=False)
        print(table)
        logging.info(f"Evaluation took {time.perf_counter() - t0:.2f} seconds")
