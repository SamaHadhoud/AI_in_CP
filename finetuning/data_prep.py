from datasets import load_dataset, concatenate_datasets
import pandas as pd

rag_dataset = load_dataset("AlaaAhmed2444/rag_with_reflection", split="train")

system_prompt = """You are a world-class competitive programmer tasked with solving a programming problem. 
You will be provided with a problem statement, and you need to create a Python3/C++ solution for it. 
Your task is to develop a winning solution to the problem.

Problem-Solving Strategy:
1. First, analyze if the problem can be solved mathematically:
   - Look for mathematical patterns or formulas that could provide direct solutions
   - Consider number theory, combinatorics, or geometric approaches
   - Check if the answer can be derived using mathematical properties instead of simulation
2. If no mathematical solution exists, then proceed with algorithmic approaches
3. For optimization problems, consider whether the answer follows a pattern or can be precomputed

Key Requirements:
1. Use the function signature: 'def solve(input_data: str) -> str:' if possible
2. Implement an algorithm that correctly solves all aspects of the problem, including edge cases
3. Optimize for both time and space complexity where possible, without compromising correctness
4. Include all necessary imports at the beginning of your code
5. Handle input and output as strings, parsing and formatting as required
6. Provide clear, concise comments explaining complex logic or optimizations

Best Practices:
- Carefully analyze the problem constraints and requirements
- Choose the most efficient algorithmic approach
- Implement robust error handling and input validation
- Use appropriate data structures to optimize complexity
- Write clean, readable code following style guidelines
- Leverage built-in functions and libraries when beneficial

Performance Optimization Guidelines:
- Avoid nested loops for large input sizes (N > 10^4)
- Consider using prefix sums, sliding windows, or hash maps
- Use list comprehensions or generator expressions where appropriate
- Leverage built-in functions like map(), filter(), zip()
- Use collections.Counter, defaultdict for counting/grouping
- Use set/dict for O(1) lookups instead of lists
- Use join() instead of += for string concatenation"""

def formatting_prompts_func_2(examples):
    conversations = []
    
    # Format RAG reflection data if available
    if 'description' in examples:
        for i in range(len(examples['description'])):
            convo = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"""Here is a competitive programming problem:

Problem Statement:
{examples['description'][i]}

Solve step by step and please provide your solution in the following format:

Solution Logic:
[Detailed analysis of the solution approach]

Solution Code:
[Implementation in Python3/C++]"""
                },
                {
                    "role": "assistant",
                    "content": f"""Let me solve this step by step:

Solution Logic:
{examples['answer_analysis'][i]}

Solution Code:
```python
{examples['cleaned_code'][i]}
```"""
                }
            ]
            conversations.append(convo)
    
    # Format HackerCup data if available
    if 'statement' in examples:
        for i in range(len(examples['statement'])):
            convo = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"""Here is a competitive programming problem:

Problem Statement:
{examples['statement'][i]}

Sample Input:
{examples['sample_input'][i]}

Sample Output:
{examples['sample_output'][i]}

Solve step by step and please provide your solution in the following format:

Solution Logic:
[Detailed analysis of the solution approach]

Solution Code:
[Implementation in Python3/C++]"""
                },
                {
                    "role": "assistant", 
                    "content": f"""Let me solve this step by step:

Solution Logic:
{examples['solution'][i]}

Implementation:
{examples['code'][i]}
"""}
            ]
            conversations.append(convo)
            
    return {"conversations": conversations}

def get_dataset():
    # Load both datasets
    rag_dataset = load_dataset("AlaaAhmed2444/rag_with_reflection", split="train")
    # hackercup_dataset = load_dataset("hackercupai/hackercup", split="full")

    # Format and combine the datasets
    formatted_rag = rag_dataset.map(
        formatting_prompts_func_2,
        remove_columns=rag_dataset.column_names,
        batched=True
    )
    return formatted_rag