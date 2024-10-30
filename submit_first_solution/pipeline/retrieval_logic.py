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
from model import call_model, count_tokens

from dataclasses import dataclass
from pathlib import Path
import logging
import simple_parsing
from mini_lib.problem24 import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run, TimeoutException

import torch
torch.cuda.empty_cache()

from models import get_vllm, get_embedding_model

language = get_language("python")
tree_parser = get_parser("python")
import multiprocessing
from typing import List, Dict
from collections import Counter
import math
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def call_model(messages):
    vllm_instance = get_vllm()
    outputs = vllm_instance.generate(messages)
    return outputs[0].outputs[0].text

# def call_default_model(messages):
#     vllm_instance = get_vllm('default')
#     outputs = vllm_instance.generate(messages)
#     return outputs[0].outputs[0].text

# def call_code_model(messages):
#     vllm_instance = get_vllm('code')
#     outputs = vllm_instance.generate(messages)
#     return outputs[0].outputs[0].text

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
    print("analysis, loookkkkk", analysis)
    return analysis

class Retriever:
    def __init__(self, path: str = "AlaaAhmed2444/rag_with_reflection"):
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
    # def index(self):
    #     corpus = self.corpus.tolist()
    #     self.corpus_tokens = [
    #         bm25s.tokenize([doc], stopwords=None)[0]
    #         for doc in corpus
    #     ]
    #     return self
    
    # def get_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
    #     """
    #     Calculate BM25 score between query and document tokens.
        
    #     Args:
    #         query_tokens: List of tokenized query terms
    #         doc_tokens: List of tokenized document terms
            
    #     Returns:
    #         float: BM25 similarity score
    #     """
    #     score = 0.0
    #     doc_len = len(doc_tokens)
        
    #     # Parameters for BM25
    #     k1 = 1.5
    #     b = 0.75
        
    #     # Calculate document frequency for each term
    #     doc_freqs = Counter(doc_tokens)
        
    #     # Calculate average document length if not already cached
    #     if not hasattr(self, '_avg_doc_len'):
    #         all_doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
    #         self._avg_doc_len = sum(all_doc_lengths) / len(all_doc_lengths) if all_doc_lengths else 0
        
    #     for term in query_tokens:
    #         if term in doc_freqs:
    #             # Term frequency in document
    #             tf = doc_freqs[term]
                
    #             # Inverse document frequency
    #             df = sum(1 for doc in self.corpus_tokens if term in doc)
    #             idf = math.log((len(self.corpus_tokens) - df + 0.5) / (df + 0.5) + 1.0)
                
    #             # BM25 score calculation
    #             numerator = tf * (k1 + 1)
    #             denominator = tf + k1 * (1 - b + b * doc_len / self._avg_doc_len)
    #             score += idf * numerator / denominator
                
    #     return score
    
    @weave.op(name="retrieve_and_rank")
    def retrieve_and_rank(self, problem, query: str, top_k: int = 3) -> List[dict]:
        logger.info(f"Starting retrieval and ranking process")
        logger.info(f"Query preview: {query[:200]}...")
        logger.info(f"Problem description preview: {problem.problem_description[:200]}...")
        
        # 1. Prepare query once
        clean_query = clean_code_string(query)
        normalized_query = normalize_code(clean_query) or clean_query
        full_query = problem.problem_description
        
        # 2. Batch process all embeddings at once
        logger.info(f"Processing embeddings for {len(self.docs)} documents...")
        all_texts = [full_query] + [f"{doc['description']}" for doc in self.docs]
        all_embeddings = get_embeddings(all_texts)
        query_embedding = all_embeddings[0]
        docs_embeddings = all_embeddings[1:]
        
        # 3. Batch process all analysis embeddings
        logger.info("Processing analysis embeddings...")
        query_analysis = self_reflection_on_problem(problem.problem_description)
        all_analyses = [query_analysis] + [doc.get('self_reflection', '') for doc in self.docs]
        all_analysis_embeddings = get_embeddings(all_analyses)
        query_analysis_embedding = all_analysis_embeddings[0]
        docs_analysis_embeddings = all_analysis_embeddings[1:]

        
        
        # 4. Calculate all similarities at once using matrix operations
        text_similarities = cosine_similarity([query_embedding], docs_embeddings)[0]
        analysis_similarities = cosine_similarity([query_analysis_embedding], docs_analysis_embeddings)[0]
        
        # 5. Calculate BM25 scores efficiently
        # logger.info("Calculating BM25 scores...")
        # try:
        #     # First tokenize the query
        #     query_tokens = bm25s.tokenize([normalized_query], stopwords=None)[0]
            
        #     # Pre-process all document tokenizations
        #     docs_tokens = [
        #         bm25s.tokenize([doc['normalized_code']], stopwords=None)[0]
        #         for doc in self.docs
        #     ]
            
        #     # Calculate BM25 scores using pre-processed tokens
        #     bm25_scores = np.array([
        #         self.retriever.get_score(query_tokens, doc_tokens)
        #         for doc_tokens in docs_tokens
        #     ])
            
        #     logger.info(f"BM25 score range: [{bm25_scores.min():.3f}, {bm25_scores.max():.3f}]")
        # except Exception as e:
        #     logger.error(f"Error in BM25 calculation: {str(e)}")
        #     bm25_scores = np.zeros(len(self.docs))
        
        # 6. Create results DataFrame
        results_dict = {
            'doc_idx': range(len(self.docs)),
            'description': [doc['description'] for doc in self.docs],
            'code': [doc['original_code'] for doc in self.docs],
            'text_similarity': text_similarities,
            'analysis_similarity': analysis_similarities,
            # 'bm25_score': bm25_scores,
            'sample_inputs': [doc['sample_inputs'] for doc in self.docs],
            'sample_outputs': [doc['sample_outputs'] for doc in self.docs],
            'answer_analysis':[doc['answer_analysis'] for doc in self.docs]
        }
        docs_df = pd.DataFrame(results_dict)
        
        # 7. Normalize scores using vectorized operations
        # logger.info("Normalizing scores...")
        # for score_col in ['text_similarity_raw', 'analysis_similarity_raw', 'bm25_score_raw']:
        #     min_score = docs_df[score_col].min()
        #     max_score = docs_df[score_col].max()
        #     if max_score != min_score:
        #         docs_df[score_col.replace('_raw', '')] = (docs_df[score_col] - min_score) / (max_score - min_score)
        #     else:
        #         docs_df[score_col.replace('_raw', '')] = np.zeros_like(docs_df[score_col])
        #     logger.debug(f"{score_col} normalized range: [{docs_df[score_col.replace('_raw', '')].min():.3f}, "
        #                 f"{docs_df[score_col.replace('_raw', '')].max():.3f}]")
        
        # 8. Calculate combined scores
        docs_df['combined_score'] = (
            0.3 * docs_df['text_similarity'] + 
            0.8 * docs_df['analysis_similarity'] 
        )
            # 0.2 * docs_df['bm25_score']
        
        # 9. Get top results efficiently
        top_docs = (docs_df
            .sort_values('combined_score', ascending=False)
            .head(top_k))
        
        # 10. Log final results
        logger.info("\nFinal score ranges:")
        for col in ['text_similarity', 'analysis_similarity', 'combined_score']:
            logger.info(f"{col}: [{docs_df[col].min():.3f}, {docs_df[col].max():.3f}]")
        
        logger.info(f"\nTop {top_k} matches:")
        for idx, row in top_docs.iterrows():
            logger.info(
                f"\nRank {idx + 1}:"
                f"\nScores: "
                # f"BM25={row['bm25_score']:.3f}, "
                f"Text={row['text_similarity']:.3f}, "
                f"Analysis={row['analysis_similarity']:.3f}, "
                f"Combined={row['combined_score']:.3f}"
                f"\nDescription preview: {row['description'][:200]}..."
            )
    
        return top_docs.to_dict(orient='records')




def get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(texts)
    
    return embeddings.tolist()


def format_examples(examples: List[dict]) -> str:
    def format_exmaple(example: dict) -> str:
        return f"""
<problem>
<problem_statement>
{example['description']}
</problem_statement>
<solution_logic>
{example['answer_analysis']}
</solution_logic>
<solution_code>
{example['code']}
</solution_code>
</problem>
"""

    messages = ""
    for idx, example in enumerate(examples):
        messages += f"\n<example{idx+1}>{format_exmaple(example)}</example{idx+1}>\n"
    return messages.strip()