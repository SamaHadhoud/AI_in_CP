
from dataclasses import dataclass
from typing import Optional, List
import weave
from mini_lib.problem import Problem
import json
from one_shot import call_model, system_prompt, extract_prompt
import logging
import re
@dataclass
class SolutionAttempt:
    code: str
    status: str
    test_cases: dict = None
    error: str = None
    execution_time: float = None

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
async def reflection(problem: Problem, solution_result: SolutionAttempt, examples) -> str:
    error_str = f"Error: {solution_result.error}" if solution_result.error else ""
    test_cases_str = json.dumps(solution_result.test_cases, indent=2) if solution_result.test_cases else ""
    Offending = ""
    if(test_cases_str):
        print(test_cases_str)
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
{"<example_that_may_help>" + examples + "</example_that_may_help>"  if examples else ""}

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
async def improve_solution(problem: Problem, previous_solution: SolutionAttempt, reflection: str = "", examples: str = "") -> str:
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
{"<example_that_may_help>" + examples + "</example_that_may_help>"  if examples else ""}
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