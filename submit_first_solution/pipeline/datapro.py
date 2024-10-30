import ast
import re
from typing import Optional
from pathlib import Path
from functools import lru_cache
import ujson as json
from datasets import load_dataset
from loguru import logger
import pandas as pd
from tree_sitter_languages import get_language, get_parser
from tqdm import tqdm
from tqdm.auto import tqdm
import numpy as np
from models.vllm_model import get_vllm
# Initialize tree-sitter parser
language = get_language("python")
tree_parser = get_parser("python")

def call_model(messages):
    vllm_instance = get_vllm()
    outputs = vllm_instance.generate(messages)
    return outputs[0].outputs[0].text


# Use the provided TOKEN_MAP
TOKEN_MAP = {
    ast.FunctionDef: "FUNC_DEF",
    ast.ClassDef: "CLASS_DEF",
    ast.BinOp: "BIN_OP",
    ast.Assign: "ASSIGN",
    ast.Expr: "EXPR",
    ast.Call: "FUNC_CALL",
    ast.If: "IF",
    ast.For: "FOR",
    ast.While: "WHILE",
    ast.Import: "IMPORT",
    ast.Return: "RETURN",
    ast.List: "LIST",
    ast.Dict: "DICT",
    ast.Name: "VAR",
    ast.Num: "NUMBER",
    ast.Constant: lambda node: (
        "NUMBER"
        if isinstance(node.value, (int, float, complex))
        else (
            "STRING"
            if isinstance(node.value, str)
            else (
                "BOOLEAN"
                if isinstance(node.value, bool)
                else "NONE" if node.value is None else "UNKNOWN"
            )
        )
    ),
}

def self_reflection_on_problem(problem):
    system_prompt = """
    You are a world-class competitive programmer tasked with analyzing programming problems. Your role is to provide a clear, concise summary of the given problem's core requirements in bullet-point format. Follow these guidelines strictly:
 
    1. Focus only on essential elements directly stated in the problem.
    2. Provide only the information explicitly stated in the problem statement.
    3. Do not infer, assume, or add any information not directly provided in the problem description.
    4. Do not attempt to solve the problem or provide solution strategies.
    5. Use the exact variable names, descriptions, units, and mathematical notation given in the problem.
    6. Include all stated constraints, even if they seem obvious.
    7. Provide only a high-level overview of what the problem asks, without adding any solution steps.
    8. If any part of the problem is unclear or ambiguous, reflect this uncertainty in your analysis.
    9. Ensure that all mathematical notations and symbols are accurately represented.
    10. Pay special attention to units (like percentages) and include them in the variable descriptions.
    11. Include any mathematical formulas or equations explicitly given in the problem statement as general rules, not specific to examples.
    12. Clearly distinguish between the general problem description and any specific examples provided.
 
    Present your analysis in a concise bullet-point format, covering the following aspects:
    - Main task or objective
    - Key variables and their descriptions
    - Constraints
    - Input format
    - Output format
    - General formulas (if any)
    - Logic flow (high-level description of what needs to be done)
    """
 
    user_prompt = """
    Analyze the following programming problem and provide a concise summary of its core requirements in bullet-point format:
 
    {problem}
 
    Remember to focus only on the essential elements explicitly stated in the problem. Do not infer or add any information not directly provided in the problem description. Be specific and use exact wording, notation, and units from the problem statement. Clearly distinguish between the general problem description and any specific examples provided.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(problem=problem)}
    ]
 
    # Call the model to get the analysis
    analysis = call_model(messages=messages)
 
    return analysis

def remove_extra_newlines(text: str) -> str:
    return re.sub(r"\n\s*\n+", "\n", text)

def remove_comments_and_docstrings(code):
    try:
        tree = tree_parser.parse(code.encode())
        root_node = tree.root_node
        
        def is_comment_or_docstring(node):
            return node.type in ('comment', 'string') and node.parent.type in ('module', 'function_definition', 'class_definition')
        
        comments_and_docstrings = [node for node in root_node.children if is_comment_or_docstring(node)]
        
        if not comments_and_docstrings:
            return code
        
        cleaned_code = []
        last_end = 0
        for node in comments_and_docstrings:
            cleaned_code.append(code[last_end:node.start_byte])
            last_end = node.end_byte
        cleaned_code.append(code[last_end:])
        
        return ''.join(cleaned_code)
    except Exception:
        return code  # Return original code if parsing fails

def clean_code_string(code: str) -> str:
    code = remove_comments_and_docstrings(code)
    code = remove_extra_newlines(code)
    return code

def tokenize_node(node):
    node_type = type(node)
    if node_type in TOKEN_MAP:
        token = TOKEN_MAP[node_type]
        if callable(token):
            yield token(node)
        else:
            yield token
    for child in ast.iter_child_nodes(node):
        yield from tokenize_node(child)

@lru_cache(maxsize=None)
def normalize_code(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
        tokens = list(tokenize_node(tree))
        return " ".join(tokens)
    except SyntaxError:
        return None

# def process_row(row):
#     description = row['description']
#     sample_inputs = row['sample_inputs']
#     sample_outputs = row['sample_outputs']
#     cf_tags = row['cf_tags'].tolist() if isinstance(row['cf_tags'], np.ndarray) else row['cf_tags']  # Convert numpy array to list
#     code = row['code']
    
#     cleaned_code = clean_code_string(code)
#     normalized_code = normalize_code(cleaned_code)
    
#     if normalized_code is not None:
#         return [{
#             "description": description,
#             "sample_inputs": sample_inputs,
#             "sample_outputs": sample_outputs,
#             "cf_tags": cf_tags,
#             "original_code": code,
#             "cleaned_code": cleaned_code,
#             "normalized_code": normalized_code
#         }]
    
#     return []  # Return an empty list if normalization fails
    
def process_row(row):
    description = row['description']
    sample_inputs = row['sample_inputs']
    sample_outputs = row['sample_outputs']
    cf_tags = row['cf_tags'].tolist() if isinstance(row['cf_tags'], np.ndarray) else row['cf_tags']
    code = row['code']
    
    cleaned_code = clean_code_string(code)
    normalized_code = normalize_code(cleaned_code)
    
    # Generate self-reflection analysis
    problem_description = f"Description: {description}\n\nSample Inputs: {sample_inputs}\n\nSample Outputs: {sample_outputs}"
    self_reflection = self_reflection_on_problem(problem_description)
    
    if normalized_code is not None:
        return [{
            "description": description,
            "sample_inputs": sample_inputs,
            "sample_outputs": sample_outputs,
            "cf_tags": cf_tags,
            "original_code": code,
            "cleaned_code": cleaned_code,
            "normalized_code": normalized_code,
            "self_reflection": self_reflection
        }]
    
    return []

# def preprocess_data(data_df: pd.DataFrame, output_path: Path, batch_size: int = 1000) -> None:
#     logger.info("Preprocessing and processing data")
    
#     with open(output_path, 'w', buffering=1024*1024) as f:
#         total_rows = len(data_df)
#         with tqdm(total=total_rows, desc="Processing rows") as pbar:
#             for _, row in data_df.iterrows():
#                 processed_row = process_row(row)
                
#                 for item in processed_row:
#                     json.dump(item, f)
#                     f.write('\n')
                
#                 pbar.update(1)

def preprocess_data(data_df: pd.DataFrame, output_path: Path, batch_size: int = 1000, append: bool = False) -> None:
    logger.info("Preprocessing and processing data")
    
    mode = 'a' if append else 'w'
    with open(output_path, mode, buffering=1024*1024) as f:
        total_rows = len(data_df)
        with tqdm(total=total_rows, desc="Processing rows") as pbar:
            for _, row in data_df.iterrows():
                processed_row = process_row(row)
                
                for item in processed_row:
                    json.dump(item, f)
                    f.write('\n')
                
                pbar.update(1)

# def get_code_contests_data(num_examples=None):
#     logger.info("Loading raw data from dataset")
#     ds = load_dataset("deepmind/code_contests", split="train")
#     if num_examples:
#         ds = ds.select(range(min(num_examples, len(ds))))

#     logger.info("Applying transformations to the dataset")
#     ds = ds.filter(lambda x: not x["is_description_translated"] and len(x["solutions"]["solution"]) > 0)
    
#     with tqdm(total=len(ds), desc="Transforming dataset") as pbar:
#         def transform_and_update(x):
#             pbar.update(1)
#             return {
#                 "description": remove_extra_newlines(x["description"]),
#                 "code": [solution for lang, solution in zip(x["solutions"]["language"], x["solutions"]["solution"]) if lang == 3],
#                 "sample_inputs": "".join(x["public_tests"]["input"]),
#                 "sample_outputs": "".join(x["public_tests"]["output"])
#             }
        
#         ds = ds.map(transform_and_update)

#     logger.info("Converting dataset to pandas DataFrame")
#     df = ds.to_pandas()
#     return df
                
def get_code_contests_data(num_examples=None):
    logger.info("Loading raw data from dataset")
    ds = load_dataset("deepmind/code_contests", split="train")
    if num_examples:
        ds = ds.select(range(min(num_examples, len(ds))))

    logger.info("Applying transformations to the dataset")
    ds = ds.filter(lambda x: len(x["solutions"]["solution"]) > 0)
    
    with tqdm(total=len(ds), desc="Transforming dataset") as pbar:
        def transform_and_update(x):
            pbar.update(1)
            python_solutions = [solution for lang, solution in zip(x["solutions"]["language"], x["solutions"]["solution"]) if lang == 3]
            return {
                "description": remove_extra_newlines(x["description"]),
                "code": python_solutions[0] if python_solutions else None,  # Take only the first Python solution
                "sample_inputs": "".join(x["public_tests"]["input"]),
                "sample_outputs": "".join(x["public_tests"]["output"]),
                "cf_tags": x["cf_tags"]  # Include the cf_tags
            }
        
        ds = ds.map(transform_and_update)

    logger.info("Filtering out examples without Python solutions")
    ds = ds.filter(lambda x: x["code"] is not None)

    logger.info("Converting dataset to pandas DataFrame")
    df = ds.to_pandas()
    return df

# if __name__ == "__main__":
#     # Get a subset of the Code Contests dataset (adjust num_examples as needed)
#     num_examples = None  # Set to None to process the entire dataset
#     data = get_code_contests_data(num_examples)
#     print(f"Loaded {len(data)} examples from the Code Contests dataset")
    
#     # Preprocess the data and save as JSONL
#     output_path = Path("/home/alaa.elsetohy/Downloads/NLP_project/project/nlp_project/submit_first_solution/pipeline/all_preprocessed_code_contests.jsonl")
#     preprocess_data(data, output_path)
#     print(f"Preprocessed data saved to {output_path}")

#     # Print the first few lines of the output file
#     print("\nSample of preprocessed data:")
#     with open(output_path, 'r') as f:
#         for _ in range(5):
#             print(json.loads(f.readline().strip()))
#             print("-" * 50)

if __name__ == "__main__":
    # Get a subset of the Code Contests dataset (adjust num_examples as needed)
    num_examples = None  # Start with a smaller number
    data = get_code_contests_data(num_examples)
    print(f"Loaded {len(data)} examples from the Code Contests dataset")
    
    # Preprocess the data and save as JSONL
    output_path = Path("/home/alaa.elsetohy/Downloads/NLP_project/project/nlp_project/submit_first_solution/pipeline/all_preprocessed_code_contests.jsonl")
    
    # Process in batches
    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        preprocess_data(batch, output_path, append=True)
        print(f"Processed batch {i//batch_size + 1}/{len(data)//batch_size + 1}")

    print(f"Preprocessed data saved to {output_path}")

    # Print the first few lines of the output file
    print("\nSample of preprocessed data:")
    with open(output_path, 'r') as f:
        for _ in range(5):
            print(json.loads(f.readline().strip()))
            print("-" * 50)