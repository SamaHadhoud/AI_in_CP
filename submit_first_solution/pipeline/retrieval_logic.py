import ast
import logging
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import warnings
import textwrap
from joblib import Parallel, delayed
import bm25s
import os
import weave
from simple_parsing import ArgumentParser
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from vllm import LLM, SamplingParams
from sklearn.feature_extraction.text import TfidfVectorizer
from tree_sitter_languages import get_language, get_parser


from dataclasses import dataclass
from pathlib import Path
import logging
import simple_parsing
from mini_lib.problem import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run, TimeoutException

import torch
torch.cuda.empty_cache()

from models import get_vllm, get_embedding_model

language = get_language("python")
tree_parser = get_parser("python")
import multiprocessing

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Token mapping for AST nodes
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
    ast.Num: "NUMBER",  # For older Python versions (< 3.8)
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
def remove_extra_newlines(text: str) -> str:
    return re.sub(r"\n\s*\n+", "\n", text)

def remove_comments_and_docstrings(code):
    doc_str_pattern = """
    (module . (expression_statement (string)) @module_doc_str)
    (class_definition body: (block . (expression_statement (string)) @class_doc_str))
    (function_definition body: (block . (expression_statement (string)) @function_doc_str))
    """
    comment_pattern = "(comment) @comment"
    tree = tree_parser.parse(code.encode())
    root_node = tree.root_node

    doc_str_query = language.query(doc_str_pattern)
    doc_strs = doc_str_query.captures(root_node)

    comment_query = language.query(comment_pattern)
    comments = comment_query.captures(root_node)

    remove_points = set((node.start_byte, node.end_byte) for node, _ in doc_strs + comments)

    cleaned_code = []
    last_index = 0
    for start, end in sorted(remove_points):
        if last_index < start:
            cleaned_code.append(code[last_index:start])
        last_index = end

    cleaned_code.append(code[last_index:])

    return "".join(cleaned_code)

def clean_code_string(code: str) -> str:
    code = remove_comments_and_docstrings(code)
    code = remove_extra_newlines(code)
    return code

def tokenize_node(node):
    node_type = type(node)
    if node_type in TOKEN_MAP:
        token = TOKEN_MAP[node_type]
        yield token(node) if callable(token) else token
    for child in ast.iter_child_nodes(node):
        yield from tokenize_node(child)

def normalize_code(code: str) -> Optional[str]:
    try:
        logger.info("Starting code normalization")
        cleaned_code = clean_code_string(code)
        logger.info(f"Cleaned code:\n{cleaned_code}")
        
        # Dedent the code to remove any leading whitespace
        dedented_code = textwrap.dedent(cleaned_code)
        logger.info(f"Dedented code:\n{dedented_code}")
        
        tree = ast.parse(dedented_code)
        tokens = list(tokenize_node(tree))
        normalized = " ".join(tokens)
        logger.info(f"Normalized code:\n{normalized}")
        return normalized
    except SyntaxError as e:
        logger.error(f"SyntaxError during normalization: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during normalization: {str(e)}")
        return None
    
def normalize_code_list(code_list: list[str]) -> list[str]:
    with multiprocessing.Pool() as pool:
        return pool.map(normalize_code, code_list)

LANGUAGE_MAP = {
    3: "Python3",
}


def clean_code(row: dict) -> dict:
    outputs = []
    for item in row["cleaned_code"]:
        item = clean_code_string(item)
        outputs.append(item)
    return {"cleaned_code": outputs}

def preprocess_data(
    input_path: Path, output_path: Path, reload_cache: bool = False
) -> pd.DataFrame:
    if output_path.exists() and not reload_cache:
        logger.info(f"Loading cached preprocessed data from {output_path}")
        return pd.read_json(output_path, lines=True)

    logger.info(f"Preprocessing data from {input_path}")
    data_df = pd.read_json(input_path, lines=True)
    data_df["normalized_code"] = normalize_code_list(data_df["cleaned_code"].tolist())
    data_df = data_df.dropna(subset=["normalized_code"])
    data_df.to_json(output_path, orient="records", lines=True)
    return data_df

class Retriever:
    def __init__(self, path: str = "param-bharat/rag-hackercup"):
        ds = load_dataset(path, split="train")
        data_df = ds.to_pandas()
        self.docs = data_df.to_dict(orient="records")
        self.corpus = data_df["normalized_code"].fillna("")  # Handle NaN values
        self.retriever = self.index()

    def index(self):
        corpus = self.corpus.tolist()
        corpus_tokens = bm25s.tokenize(corpus, stopwords=None)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        return retriever

    @weave.op(name="retrieve_docs")
    def retrieve(self, query: str, k: int = 10):
        logger.info(f"Original query:\n{query}")
        
        clean_query = clean_code_string(query)
        logger.info(f"Cleaned query:\n{clean_query}")
        
        normalized_query = normalize_code(clean_query)
        logger.info(f"Normalized query:\n{normalized_query}")
        
        if normalized_query is None:
            logger.warning("Failed to normalize query. Using cleaned query.")
            normalized_query = clean_query
        
        query_tokens = bm25s.tokenize([normalized_query], stopwords=None)
        logger.info(f"Query tokens: {query_tokens}")
        
        try:
            docs, _ = self.retriever.retrieve(query_tokens[0], k=k, corpus=self.docs)
            return docs[0, :].tolist()
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []
        
# model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.95)

def get_embeddings(texts, vectorizer=None):
    vllm = get_vllm()
    if isinstance(texts, str):
        texts = [texts]
    
    prompt_template = "Summarize the following code in one sentence: {}"
    prompts = [prompt_template.format(text) for text in texts]
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    outputs = vllm.generate(prompts, sampling_params)
    
    summaries = [output.outputs[0].text.strip() for output in outputs]
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(summaries)
    else:
        embeddings = vectorizer.transform(summaries)
    
    return vectorizer, embeddings.toarray()

def rerank_docs(problem, query: str, retrieved_docs: List[dict], top_k: int = 3) -> List[dict]:
    print(f"Number of retrieved docs: {len(retrieved_docs)}")
    
    _,query_embeddings = get_embeddings(
        problem.problem_description + " " + query
    )
    print(f"Shape of query_embeddings: {query_embeddings.shape}")
    
    docs_embeddings = []
    for doc in retrieved_docs:
        _, doc_embedding = get_embeddings(doc["description"] + " " + doc["cleaned_code"])
        docs_embeddings.append(doc_embedding[0])  # Assuming each embedding is a 1D array
    docs_embeddings = np.array(docs_embeddings)
    
    print(f"Shape of docs_embeddings: {docs_embeddings.shape}")

    similarities = cosine_similarity(query_embeddings, docs_embeddings)
    docs_df = pd.DataFrame(retrieved_docs)
    docs_df["similarity"] = similarities[0]
    docs_df = docs_df.sort_values(by="similarity", ascending=False)
    docs_df = docs_df.drop_duplicates(
        subset=["description"],
        keep="first",
    )
    return docs_df.head(top_k).to_dict(orient="records")

    # if not retrieved_docs:
    #     return []
    
    # vectorizer, docs_embeddings = get_embeddings([doc["cleaned_code"] for doc in retrieved_docs])
    # _, query_embeddings = get_embeddings(query, vectorizer)

    
    # similarities = cosine_similarity(query_embeddings, docs_embeddings).flatten()
    # docs_df = pd.DataFrame(retrieved_docs)
    # docs_df["similarity"] = similarities
    # docs_df = docs_df.sort_values(by="similarity", ascending=False)
    # docs_df = docs_df.drop_duplicates(subset=["cleaned_code"], keep="first")
    # return docs_df.head(top_k).to_dict(orient="records")


def format_examples(examples: List[dict]) -> str:
    def format_exmaple(example: dict) -> str:
        return f"""
<problem>
<problem_statement>
{example['description']}
</problem_statement>
<source_code>
{example['cleaned_code']}
</source_code>
</problem>
"""

    messages = ""
    for example in examples:
        messages += f"\n<example>{format_exmaple(example)}</example>\n"
    return messages.strip()