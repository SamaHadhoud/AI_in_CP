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

# Import the solve_problem function and other necessary components
# from one_shot import solve_problem, system_prompt, prompt_template, extract_prompt
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
    for item in row["code"]:
        item = clean_code_string(item)
        outputs.append(item)
    return {"code": outputs}

def preprocess_data(
    input_path: Path, output_path: Path, reload_cache: bool = False
) -> pd.DataFrame:
    if output_path.exists() and not reload_cache:
        logger.info(f"Loading cached preprocessed data from {output_path}")
        return pd.read_json(output_path, lines=True)

    logger.info(f"Preprocessing data from {input_path}")
    data_df = pd.read_json(input_path, lines=True)
    data_df["normalized_code"] = normalize_code_list(data_df["code"].tolist())
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

def rerank_docs(query: str, retrieved_docs: List[dict], top_k: int = 3) -> List[dict]:
    if not retrieved_docs:
        return []
    
    vectorizer, docs_embeddings = get_embeddings([doc["code"] for doc in retrieved_docs])
    _, query_embeddings = get_embeddings(query, vectorizer)

    similarities = cosine_similarity(query_embeddings, docs_embeddings)[0]
    docs_df = pd.DataFrame(retrieved_docs)
    docs_df["similarity"] = similarities
    docs_df = docs_df.sort_values(by="similarity", ascending=False)
    docs_df = docs_df.drop_duplicates(subset=["code"], keep="first")
    return docs_df.head(top_k).to_dict(orient="records")

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("-c", "--cache-directory", type=Path, default="data/cache")
#     parser.add_argument("--reload-cache", action="store_true")
#     parser.add_argument("--problem-name", type=str, default="dim_sum_delivery")
#     parser.add_argument("--folder-path", type=Path, default=Path("./dataset/2023/practice/"))
#     parser.add_argument("--use-images", action="store_true")
#     parser.add_argument("--timeout", type=int, default=60)
#     parser.add_argument("--save-output", action="store_true", default=True)

#     args = parser.parse_args()

#     if not args.cache_directory.exists():
#         args.cache_directory.mkdir(parents=True)

#     retriever = Retriever()

#     # Load the problem
#     problem = Problem.from_name(args.problem_name, args.folder_path)

#     # Generate code using zero-shot approach
#     try:
#         problem_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout)
#         generated_code = problem_solution["code"]
#     except TimeoutException:
#         print("The solution took too long to execute and was terminated.")
#         generated_code = "# Failed to generate code due to timeout"

#     # Use the generated code as a query for retrieval
#     try:
#         # Retrieve documents
#         print("\nAttempting to retrieve documents...")
#         retrieved_docs = retriever.retrieve(generated_code, k=5)

#         if not retrieved_docs:
#             print("No documents retrieved. Check the query processing.")
#         else:
#             print("\nRetrieved Documents:")
#             for i, doc in enumerate(retrieved_docs, 1):
#                 print(f"\nDocument {i}:")
#                 print(f"Code:\n{doc['code'][:200]}...")  # Print first 200 characters of the code
#                 print(f"Description: {doc['description'][:100]}...")  # Print first 100 characters of the description

#         # Rerank documents
#         # Rerank documents
#         reranked_docs = rerank_docs(generated_code, retrieved_docs, top_k=3)

#         if not reranked_docs:
#             print("No documents after reranking. Check the reranking process.")
#         else:
#             print("\nReranked Documents:")
#             for i, doc in enumerate(reranked_docs, 1):
#                 # Remove 'normalized_code' from each document
#                 doc.pop('normalized_code', None)
                
#                 print(f"\nDocument {i}:")
#                 print(f"Description:\n{doc['description']}")
#                 print(f"\nCode:\n{doc['code']}")
#                 print(f"Similarity: {doc['similarity']:.4f}")

#         try:
#             # Prepare examples for the prompt
#             examples = """Examples:
#         """
#             for i, doc in enumerate(reranked_docs, 1):
#                 examples += f"""Example {i}:
#         Description:
#         {doc['description']}

#         Code:
#         {doc['code']}

#         """
#             problem_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout, examples=examples)
#             generated_code = problem_solution["code"]
#         except TimeoutException:
#             print("The solution took too long to execute and was terminated.")
#             generated_code = "# Failed to generate code due to timeout"

#         matches = check_solution(problem_solution["expected_output"], problem_solution["generated_output"])
#         logging.info("Sample Matches:")
#         logging.info(matches)

#         logging.info("> Solving on full input...")
#         expected_output = problem.get_output()
#         generated_output = run(problem_solution["code"], input=problem.get_input(), timeout=args.timeout) 
#         matches = check_solution(expected_output, generated_output)
#         logging.info("Final Matches:")
#         logging.info(matches)

#         if args.save_output:
#             logging.info("> Saving output to files")
#             problem.save_output(problem_solution["generated_output"])
#             problem.save_code(problem_solution["code"])

            ###########

        # print("\nReranking documents...")
        # reranked_docs = rerank_docs(generated_code, retrieved_docs, top_k=3)

        # if not reranked_docs:
        #     print("No documents after reranking. Check the reranking process.")
        # else:
        #     print("\nReranked Documents:")
        #     for i, doc in enumerate(reranked_docs, 1):
        #         # Remove 'normalized_code' from each document
        #         doc.pop('normalized_code', None)
                
        #         print(f"\nDocument {i}:")
        #         print(f"Code:\n{doc['code'][:200]}...")  # Print first 200 characters of the code
        #         print(f"Description: {doc['description'][:100]}...")  # Print first 100 characters of the description
        #         print(f"Similarity: {doc['similarity']:.4f}")

        # try:
        #     problem_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout, examples=str(reranked_docs))
        #     generated_code = problem_solution["code"]
        # except TimeoutException:
        #     print("The solution took too long to execute and was terminated.")
        #     generated_code = "# Failed to generate code due to timeout"
        # print("\nReranking documents...")
        # reranked_docs = rerank_docs(generated_code, retrieved_docs, top_k=3)

        # if not reranked_docs:
        #     print("No documents after reranking. Check the reranking process.")
        # else:
        #     print("\nReranked Documents:")
        #     for i, doc in enumerate(reranked_docs, 1):
        #         print(f"\nDocument {i}:")
        #         print(f"Code:\n{doc['code'][:200]}...")  # Print first 200 characters of the code
        #         print(f"Description: {doc['description'][:100]}...")  # Print first 100 characters of the description
        #         print(f"Similarity: {doc['similarity']:.4f}")

        # try:
        #     problem_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout, examples=str(reranked_docs))
        #     generated_code = problem_solution["code"]
        # except TimeoutException:
        #     print("The solution took too long to execute and was terminated.")
        #     generated_code = "# Failed to generate code due to timeout"


        

    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("-c", "--cache-directory", type=Path, default="data/cache")
#     parser.add_argument("--reload-cache", action="store_true")

#     args = parser.parse_args()

#     if not args.cache_directory.exists():
#         args.cache_directory.mkdir(parents=True)

#     retriever = Retriever()

#     # Test query
#     test_query = """
#     def factorial(n):
#         if n == 0:
#             return 1
#         else:
#             return n * factorial(n-1)
#     """
#     try:
#         # Retrieve documents
#         print("\nAttempting to retrieve documents...")
#         retrieved_docs = retriever.retrieve(test_query, k=5)

#         if not retrieved_docs:
#             print("No documents retrieved. Check the query processing.")
#         else:
#             print("\nRetrieved Documents:")
#             for i, doc in enumerate(retrieved_docs, 1):
#                 print(f"\nDocument {i}:")
#                 print(f"Code:\n{doc['code'][:200]}...")  # Print first 200 characters of the code
#                 print(f"Description: {doc['description'][:100]}...")  # Print first 100 characters of the description

#         # Rerank documents
#         print("\nReranking documents...")
#         reranked_docs = rerank_docs(test_query, retrieved_docs, top_k=3)

#         if not reranked_docs:
#             print("No documents after reranking. Check the reranking process.")
#         else:
#             print("\nReranked Documents:")
#             for i, doc in enumerate(reranked_docs, 1):
#                 print(f"\nDocument {i}:")
#                 print(f"Code:\n{doc['code'][:200]}...")  # Print first 200 characters of the code
#                 print(f"Description: {doc['description'][:100]}...")  # Print first 100 characters of the description
#                 print(f"Similarity: {doc['similarity']:.4f}")

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         import traceback
#         traceback.print_exc()






# import ast
# import logging
# from pathlib import Path
# from typing import List, Optional
# from datasets import load_dataset
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# import re
# import warnings
# import textwrap
# from joblib import Parallel, delayed

# logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Token mapping for AST nodes
# TOKEN_MAP = {
#     ast.FunctionDef: "FUNC_DEF",
#     ast.ClassDef: "CLASS_DEF",
#     ast.BinOp: "BIN_OP",
#     ast.Assign: "ASSIGN",
#     ast.Expr: "EXPR",
#     ast.Call: "FUNC_CALL",
#     ast.If: "IF",
#     ast.For: "FOR",
#     ast.While: "WHILE",
#     ast.Import: "IMPORT",
#     ast.Return: "RETURN",
#     ast.List: "LIST",
#     ast.Dict: "DICT",
#     ast.Name: "VAR",
#     ast.Num: "NUMBER",
#     ast.Constant: lambda node: (
#         "NUMBER" if isinstance(node.value, (int, float, complex))
#         else "STRING" if isinstance(node.value, str)
#         else "BOOLEAN" if isinstance(node.value, bool)
#         else "NONE" if node.value is None
#         else "UNKNOWN"
#     ),
# }

# def tokenize_node(node):
#     """Tokenizes an AST node using the TOKEN_MAP dictionary."""
#     node_type = type(node)
#     if node_type in TOKEN_MAP:
#         token = TOKEN_MAP[node_type]
#         if callable(token):
#             yield token(node)
#         else:
#             yield token
#     for child in ast.iter_child_nodes(node):
#         yield from tokenize_node(child)

# def normalize_code(code: str) -> Optional[str]:
#     """Tokenizes and normalizes any Python code snippet."""
#     try:
#         code = textwrap.dedent(code).strip()
#         tree = ast.parse(code)
#     except SyntaxError:
#         logger.warning(f"Failed to parse code: {code[:50]}...")
#         return None
#     tokens = list(tokenize_node(tree))
#     return " ".join(tokens)

# def normalize_code_list(code_list: List[str]) -> List[Optional[str]]:
#     return Parallel(n_jobs=-1)(delayed(normalize_code)(code) for code in code_list)

# def clean_code_string(code: str) -> str:
#     code = re.sub(r'#.*', '', code)
#     code = re.sub(r'\s+', ' ', code).strip()
#     return code

# class BM25:
#     def __init__(self, corpus, k1=1.5, b=0.75):
#         self.corpus = corpus
#         self.k1 = k1
#         self.b = b
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
#             self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
#             self.doc_term_matrix = self.vectorizer.fit_transform(corpus)
#         self.idf = self._compute_idf()
#         self.doc_len = np.asarray(self.doc_term_matrix.sum(axis=1)).flatten()
#         self.avg_doc_len = self.doc_len.mean()

#     def _compute_idf(self):
#         df = np.bincount(self.doc_term_matrix.indices, minlength=self.doc_term_matrix.shape[1])
#         return np.log((self.doc_term_matrix.shape[0] - df + 0.5) / (df + 0.5) + 1)

#     def get_scores(self, query):
#         query_vec = self.vectorizer.transform([query])
#         scores = np.zeros(self.doc_term_matrix.shape[0])
#         for term, freq in zip(query_vec.indices, query_vec.data):
#             qtf = np.sqrt(freq)
#             d_tf = self.doc_term_matrix[:, term].toarray().flatten()
#             scores += (self.idf[term] * qtf * d_tf * (self.k1 + 1) /
#                        (d_tf + self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_doc_len)))
#         return scores

#     def retrieve(self, query, k=10):
#         scores = self.get_scores(query)
#         top_indices = np.argsort(scores)[::-1][:k]
#         return top_indices, scores[top_indices]

# class Retriever:
#     def __init__(self, path: str = "param-bharat/rag-hackercup"):
#         try:
#             ds = load_dataset(path, split="train")
#             data_df = ds.to_pandas()
#             self.docs = data_df.to_dict(orient="records")
#             self.corpus = data_df["normalized_code"].tolist()
#             self.retriever = self.index()
#             logger.info(f"Loaded {len(self.docs)} documents from {path}")
#         except Exception as e:
#             logger.error(f"Failed to initialize Retriever: {str(e)}")
#             raise

#     def index(self):
#         return BM25(self.corpus)

#     def retrieve(self, query: str, k: int = 10):
#         clean_query = clean_code_string(query)
#         normalized_query = normalize_code(clean_query)
#         if normalized_query is None:
#             logger.warning("Failed to normalize query, using cleaned query instead")
#             normalized_query = clean_query
#         try:
#             indices, _ = self.retriever.retrieve(normalized_query, k=k)
#             return [self.docs[i] for i in indices]
#         except Exception as e:
#             logger.error(f"Error during retrieval: {str(e)}")
#             return []

# def preprocess_data(data_df: pd.DataFrame) -> pd.DataFrame:
#     logger.info("Preprocessing data")
#     data_df["normalized_code"] = normalize_code_list(data_df["code"].tolist())
#     data_df = data_df.dropna(subset=["normalized_code"])
#     return data_df

# # Example usage
# if __name__ == "__main__":
#     try:
#         # Load and preprocess data
#         ds = load_dataset("param-bharat/rag-hackercup", split="train")
#         data_df = ds.to_pandas()
#         preprocessed_df = preprocess_data(data_df)
        
#         # Initialize retriever
#         retriever = Retriever()
        
#         # Example query
#         query = """
#         def fibonacci(n):
#             if n <= 1:
#                 return n
#             else:
#                 return fibonacci(n-1) + fibonacci(n-2)
#         """
#         results = retriever.retrieve(query, k=5)
        
#         if results:
#             for i, doc in enumerate(results, 1):
#                 print(f"Result {i}:")
#                 print(f"Description: {doc['description'][:100]}...")
#                 print(f"Code: {doc['code'][:100]}...")
#                 print()
#         else:
#             print("No results found.")
        
#         # Print normalized query for debugging
#         print("Normalized Query:")
#         print(normalize_code(query))
        
#     except Exception as e:
#         logger.error(f"An error occurred: {str(e)}")


#####################################################################
# import ast
# import logging
# from pathlib import Path
# from typing import List, Optional
# from datasets import load_dataset
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# import re
# import warnings
# import textwrap
# from joblib import Parallel, delayed
# import bm25s

# import ast
# import logging
# import os
# from pathlib import Path
# from typing import List, Optional

# import bm25s
# import pandas as pd
# import weave
# from datasets import load_dataset
# from joblib import Parallel, delayed
# from openai import AsyncOpenAI
# from simple_parsing import ArgumentParser
# from sklearn.metrics.pairwise import cosine_similarity

# from transformers import AutoTokenizer, AutoModel
# import torch
# from vllm import LLM, SamplingParams
# from sklearn.feature_extraction.text import TfidfVectorizer

# logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Token mapping for AST nodes
# TOKEN_MAP = {
#     ast.FunctionDef: "FUNC_DEF",
#     ast.ClassDef: "CLASS_DEF",
#     ast.BinOp: "BIN_OP",
#     ast.Assign: "ASSIGN",
#     ast.Expr: "EXPR",
#     ast.Call: "FUNC_CALL",
#     ast.If: "IF",
#     ast.For: "FOR",
#     ast.While: "WHILE",
#     ast.Import: "IMPORT",
#     ast.Return: "RETURN",
#     ast.List: "LIST",
#     ast.Dict: "DICT",
#     ast.Name: "VAR",
#     ast.Num: "NUMBER",  # For older Python versions (< 3.8)
#     ast.Constant: lambda node: (
#         "NUMBER"
#         if isinstance(node.value, (int, float, complex))
#         else (
#             "STRING"
#             if isinstance(node.value, str)
#             else (
#                 "BOOLEAN"
#                 if isinstance(node.value, bool)
#                 else "NONE" if node.value is None else "UNKNOWN"
#             )
#         )
#     ),
# }

# def tokenize_node(node):
#     """Tokenizes an AST node using the TOKEN_MAP dictionary."""
#     node_type = type(node)
#     if node_type in TOKEN_MAP:
#         token = TOKEN_MAP[node_type]
#         if callable(token):
#             yield token(node)
#         else:
#             yield token
#     for child in ast.iter_child_nodes(node):
#         yield from tokenize_node(child)

# def normalize_code(code: str) -> Optional[str]:
#     """Tokenizes and normalizes any Python code snippet."""
#     try:
#         code = textwrap.dedent(code).strip()
#         tree = ast.parse(code)
#     except SyntaxError:
#         logger.warning(f"Failed to parse code: {code[:50]}...")
#         return None
#     tokens = list(tokenize_node(tree))
#     return " ".join(tokens)

# def normalize_code_list(code_list: List[str]) -> List[Optional[str]]:
#     return Parallel(n_jobs=-1)(delayed(normalize_code)(code) for code in code_list)

# def clean_code_string(code: str) -> str:
#     code = re.sub(r'#.*', '', code)
#     code = re.sub(r'\s+', ' ', code).strip()
#     return code

# # def tokenize_node(node):
# #     """Tokenizes an AST node using the TOKEN_MAP dictionary."""
# #     node_type = type(node)

# #     # Handle the case where the node type is in the TOKEN_MAP
# #     if node_type in TOKEN_MAP:
# #         token = TOKEN_MAP[node_type]
# #         if callable(
# #             token
# #         ):  # If the token is a function (for complex cases like ast.Constant)
# #             yield token(node)
# #         else:
# #             yield token

#     # # Recursively process child nodes
#     # for child in ast.iter_child_nodes(node):
#     #     yield from tokenize_node(child)


# # def normalize_code(code: str) -> Optional[str]:
# #     """Tokenizes and normalizes any Python code snippet."""
# #     try:
# #         tree = ast.parse(code)
# #     except SyntaxError as e:
# #         return None

# #     tokens = list(tokenize_node(tree))
# #     return " ".join(tokens)


# # def normalize_code_list(code_list: list[str]) -> list[str]:
# #     if len(code_list) > 1000:
# #         return Parallel(n_jobs=-1)(delayed(normalize_code)(code) for code in code_list)
# #     else:
# #         return [normalize_code(code) for code in code_list]


# def preprocess_data(
#     input_path: Path, output_path: Path, reload_cache: bool = False
# ) -> pd.DataFrame:
#     if output_path.exists() and not reload_cache:
#         logger.info(f"Loading cached preprocessed data from {output_path}")
#         return pd.read_json(output_path, lines=True)

#     logger.info(f"Preprocessing data from {input_path}")
#     data_df = pd.read_json(input_path, lines=True)
#     data_df["normalized_code"] = normalize_code_list(data_df["code"].tolist())
#     data_df = data_df.dropna(subset=["normalized_code"])
#     data_df.to_json(output_path, orient="records", lines=True)
#     return data_df


# class Retriever:
#     def __init__(self, path: str = "param-bharat/rag-hackercup"):
#         ds = load_dataset(path, split="train")
#         data_df = ds.to_pandas()
#         self.docs = data_df.to_dict(orient="records")
#         self.corpus = data_df["normalized_code"]
#         self.retriever = self.index()

#     def index(self):
#         corpus = self.corpus.tolist()
#         corpus_tokens = bm25s.tokenize(corpus, stopwords=None)
#         retriever = bm25s.BM25(corpus=corpus)
#         retriever.index(corpus_tokens)
#         return retriever

#     @weave.op(name="retrieve_docs")
#     def retrieve(self, query: str, k: int = 10):
#         clean_query = clean_code_string(query)
#         normalized_query = normalize_code(clean_query)
#         query_tokens = bm25s.tokenize(normalized_query, stopwords=None)
#         docs, _ = self.retriever.retrieve(query_tokens, k=k, corpus=self.docs)
#         return docs[0, :].tolist()


# def index_data(
#     input_path: Path,
#     output_path: Path,
#     reload_cache: bool = False,
# ):
#     if output_path.exists() and not reload_cache:
#         logger.info(f"Loading cached retriever from {output_path}")
#         return Retriever.load(output_path)
#     logger.info(f"Creating retriever from {input_path}")
#     data_df = pd.read_json(input_path, lines=True, orient="records")
#     retriever = Retriever(data_df=data_df)
#     retriever.index()
#     retriever.save(output_path)
#     return retriever

# model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.95)\


# def get_embeddings(texts, vectorizer=None):
#     if isinstance(texts, str):
#         texts = [texts]
    
#     # Use vLLM to generate a summary or representation of each text
#     prompt_template = "Summarize the following code in one sentence: {}"
#     prompts = [prompt_template.format(text) for text in texts]
    
#     sampling_params = SamplingParams(temperature=0.0, max_tokens=50)  # Adjust max_tokens as needed
#     outputs = llm.generate(prompts, sampling_params)
    
#     # Extract the generated summaries
#     summaries = [output.outputs[0].text.strip() for output in outputs]
    
#     # Use TF-IDF to create embeddings from the summaries
#     if vectorizer is None:
#         vectorizer = TfidfVectorizer()
#         embeddings = vectorizer.fit_transform(summaries)
#     else:
#         embeddings = vectorizer.transform(summaries)
    
#     return vectorizer, embeddings.toarray()

# def rerank_docs(query: str, retrieved_docs: List[dict], top_k: int = 3) -> List[dict]:
#     # First, get embeddings for documents
#     vectorizer, docs_embeddings = get_embeddings([doc["code"] for doc in retrieved_docs])
    
#     # Then, use the same vectorizer for the query
#     _, query_embeddings = get_embeddings(query, vectorizer)

#     similarities = cosine_similarity(query_embeddings, docs_embeddings)[0]
#     docs_df = pd.DataFrame(retrieved_docs)
#     docs_df["similarity"] = similarities
#     docs_df = docs_df.sort_values(by="similarity", ascending=False)
#     docs_df = docs_df.drop_duplicates(subset=["code"], keep="first")
#     return docs_df.head(top_k).to_dict(orient="records")


# # @weave.op(name="get_embeddings")
# # async def get_embeddings(texts, model=EMBEDDING_MODEL):
# #     async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
# #     if isinstance(texts, str):
# #         texts = [texts]
# #     texts = [text.replace("\n", " ") for text in texts]
# #     response = await async_client.embeddings.create(
# #         input=texts, model=model, dimensions=512
# #     )
# #     return [embedding.embedding for embedding in response.data]


# # @weave.op(name="rerank_docs")
# # async def rerank_docs(
# #     problem: Problem,
# #     solution: Solution,
# #     retrieved_docs: List[dict],
# #     top_k: int = 3,
# # ) -> List[dict]:
# #     query_embeddings = await get_embeddings(
# #         problem.problem_description + " " + solution.source_code
# #     )
# #     docs_embeddings = await get_embeddings(
# #         [doc["description"] + " " + doc["code"] for doc in retrieved_docs]
# #     )

# #     similarities = cosine_similarity(query_embeddings, docs_embeddings)
# #     docs_df = pd.DataFrame(retrieved_docs)
# #     docs_df["similarity"] = similarities[0]
# #     docs_df = docs_df.sort_values(by="similarity", ascending=False)
# #     docs_df = docs_df.drop_duplicates(
# #         subset=["description"],
# #         keep="first",
# #     )
# #     return docs_df.head(top_k).to_dict(orient="records")


# if __name__ == "__main__":

#     parser = ArgumentParser()
#     parser.add_argument("-c", "--cache-directory", type=Path, default="data/cache")
#     parser.add_argument("--reload-cache", action="store_true")

#     args = parser.parse_args()

#     if not args.cache_directory.exists():
#         args.cache_directory.mkdir(parents=True)

#     if (args.cache_directory / "retriever").exists():
#         retriever = Retriever.load(args.cache_directory / "retriever")
#     elif (args.cache_directory / "preprocessed.jsonl").exists():
#         preprocessed_df = preprocess_data(
#             args.cache_directory / "raw.jsonl",
#             args.cache_directory / "preprocessed.jsonl",
#             args.reload_cache,
#         )
#         retriever = Retriever(data_df=preprocessed_df)
#         retriever.index()
#         retriever.save(args.cache_directory / "retriever")
#     else:
#         raw_df = get_code_contests_data(
#             args.cache_directory / "raw.jsonl", args.reload_cache
#         )
#         preprocessed_df = preprocess_data(
#             args.cache_directory / "raw.jsonl",
#             args.cache_directory / "preprocessed.jsonl",
#             args.reload_cache,
#         )
#         retriever = Retriever(data_df=preprocessed_df)
#         retriever.index()
#         retriever.save(args.cache_directory / "retriever")

# #


# import ast
# import logging
# from pathlib import Path
# from typing import List, Optional
# from datasets import load_dataset
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# import re
# import warnings
# import textwrap
# from joblib import Parallel, delayed
# from simple_parsing import ArgumentParser
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoTokenizer, AutoModel
# import torch
# from vllm import LLM, SamplingParams
# from sklearn.feature_extraction.text import TfidfVectorizer

# logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Token mapping for AST nodes
# TOKEN_MAP = {
#     ast.FunctionDef: "FUNC_DEF",
#     ast.ClassDef: "CLASS_DEF",
#     ast.BinOp: "BIN_OP",
#     ast.Assign: "ASSIGN",
#     ast.Expr: "EXPR",
#     ast.Call: "FUNC_CALL",
#     ast.If: "IF",
#     ast.For: "FOR",
#     ast.While: "WHILE",
#     ast.Import: "IMPORT",
#     ast.Return: "RETURN",
#     ast.List: "LIST",
#     ast.Dict: "DICT",
#     ast.Name: "VAR",
#     ast.Num: "NUMBER",
#     ast.Constant: lambda node: (
#         "NUMBER" if isinstance(node.value, (int, float, complex))
#         else "STRING" if isinstance(node.value, str)
#         else "BOOLEAN" if isinstance(node.value, bool)
#         else "NONE" if node.value is None
#         else "UNKNOWN"
#     ),
# }
# def remove_extra_newlines(text: str) -> str:
#     # Use regex to replace 2 or more newlines (with possible whitespace in between) with a single newline
#     text = re.sub(r"\n\s*\n+", "\n", text)
#     return text

# LANGUAGE_MAP = {
#     3: "Python3",
# }


# def clean_code(row: dict) -> dict:
#     outputs = []
#     for item in row["code"]:
#         item = clean_code_string(item)
#         outputs.append(item)
#     return {"code": outputs}


# def get_solution(row: dict) -> dict:
#     solutions = row["solutions"]
#     languages = solutions["language"]
#     solutions = solutions["solution"]

#     outputs = []
#     for language, solution in zip(languages, solutions):
#         language = LANGUAGE_MAP.get(language)
#         if language:
#             outputs.append(solution)
#     return {"code": outputs}


# def get_test_cases(row: dict) -> dict:
#     tests = row["public_tests"]
#     return {
#         "sample_inputs": "".join(tests["input"]),
#         "sample_outputs": "".join(tests["output"]),
#     }


# def clean_description(row: dict) -> dict:
#     description = row["description"]
#     description = remove_extra_newlines(description)
#     return {"description": description}


# def get_code_contests_data(cache_file: Path, reload_cache: bool = False):
#     if cache_file.exists() and not reload_cache:
#         logger.info(f"Loading cached raw data from {cache_file}")
#         return pd.read_json(cache_file, lines=True)

#     logger.info(f"Loading raw data from dataset")
#     ds = load_dataset("deepmind/code_contests")

#     train_ds = ds["train"].map(get_solution, num_proc=4)
#     train_ds = train_ds.filter(lambda x: not x["is_description_translated"], num_proc=4)
#     train_ds = train_ds.filter(lambda x: len(x["code"]) > 0, num_proc=4)
#     train_ds = train_ds.map(clean_code, num_proc=4)
#     train_ds = train_ds.map(clean_description, num_proc=4)
#     train_ds = train_ds.map(get_test_cases, num_proc=4)
#     train_ds = train_ds.remove_columns(
#         [
#             col
#             for col in train_ds.column_names
#             if col not in ["description", "code", "sample_inputs", "sample_outputs"]
#         ]
#     )

#     train_df = train_ds.to_pandas()
#     train_df = train_df.explode("code").reset_index(drop=True)
#     train_df = train_df.drop_duplicates(subset=["code"], keep="first")
#     train_df.to_json(cache_file, orient="records", lines=True)
#     return train_df

# def tokenize_node(node):
#     """Tokenizes an AST node using the TOKEN_MAP dictionary."""
#     node_type = type(node)
#     if node_type in TOKEN_MAP:
#         token = TOKEN_MAP[node_type]
#         if callable(token):
#             yield token(node)
#         else:
#             yield token
#     for child in ast.iter_child_nodes(node):
#         yield from tokenize_node(child)

# def normalize_code(code: str) -> Optional[str]:
#     """Tokenizes and normalizes any Python code snippet."""
#     try:
#         tree = ast.parse(code)
#     except SyntaxError as e:
#         return None

#     tokens = list(tokenize_node(tree))
#     return " ".join(tokens)


# def normalize_code_list(code_list: list[str]) -> list[str]:
#     if len(code_list) > 1000:
#         return Parallel(n_jobs=-1)(delayed(normalize_code)(code) for code in code_list)
#     else:
#         return [normalize_code(code) for code in code_list]

# def clean_code_string(code: str) -> str:
#     code = re.sub(r'#.*', '', code)
#     code = re.sub(r'\s+', ' ', code).strip()
#     return code

# def preprocess_data(
#     input_path: Path, output_path: Path, reload_cache: bool = False
# ) -> pd.DataFrame:
#     if output_path.exists() and not reload_cache:
#         logger.info(f"Loading cached preprocessed data from {output_path}")
#         return pd.read_json(output_path, lines=True)

#     logger.info(f"Preprocessing data from {input_path}")
#     data_df = pd.read_json(input_path, lines=True)
#     data_df["normalized_code"] = normalize_code_list(data_df["code"].tolist())
#     data_df = data_df.dropna(subset=["normalized_code"])
#     data_df.to_json(output_path, orient="records", lines=True)
#     return data_df

# class BM25:
#     def __init__(self, corpus, k1=1.5, b=0.75):
#         self.corpus = corpus
#         self.k1 = k1
#         self.b = b
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
#             self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
#             self.doc_term_matrix = self.vectorizer.fit_transform(corpus)
#         self.idf = self._compute_idf()
#         self.doc_len = np.asarray(self.doc_term_matrix.sum(axis=1)).flatten()
#         self.avg_doc_len = self.doc_len.mean()

#     def _compute_idf(self):
#         df = np.bincount(self.doc_term_matrix.indices, minlength=self.doc_term_matrix.shape[1])
#         return np.log((self.doc_term_matrix.shape[0] - df + 0.5) / (df + 0.5) + 1)

#     def get_scores(self, query):
#         query_vec = self.vectorizer.transform([query])
#         scores = np.zeros(self.doc_term_matrix.shape[0])
#         for term, freq in zip(query_vec.indices, query_vec.data):
#             qtf = np.sqrt(freq)
#             d_tf = self.doc_term_matrix[:, term].toarray().flatten()
#             scores += (self.idf[term] * qtf * d_tf * (self.k1 + 1) /
#                        (d_tf + self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_doc_len)))
#         return scores

#     def retrieve(self, query, k=10):
#         scores = self.get_scores(query)
#         top_indices = np.argsort(scores)[::-1][:k]
#         return top_indices, scores[top_indices]

# class Retriever:
#     def __init__(self, path: str = "param-bharat/rag-hackercup"):
#         try:
#             ds = load_dataset(path, split="train")
#             data_df = ds.to_pandas()
#             self.docs = data_df.to_dict(orient="records")
#             self.corpus = data_df["normalized_code"].tolist()
#             self.retriever = self.index()
#             logger.info(f"Loaded {len(self.docs)} documents from {path}")
#         except Exception as e:
#             logger.error(f"Failed to initialize Retriever: {str(e)}")
#             raise

#     def index(self):
#         return BM25(self.corpus)

#     def retrieve(self, query: str, k: int = 10):
#         clean_query = clean_code_string(query)
#         normalized_query = normalize_code(clean_query)
#         if normalized_query is None:
#             logger.warning("Failed to normalize query, using cleaned query instead")
#             normalized_query = clean_query
#         try:
#             indices, _ = self.retriever.retrieve(normalized_query, k=k)
#             return [self.docs[i] for i in indices]
#         except Exception as e:
#             logger.error(f"Error during retrieval: {str(e)}")
#             return []

#     def save(self, path: Path):
#         # Implement save functionality if needed
#         pass

#     @classmethod
#     def load(cls, path: Path):
#         # Implement load functionality if needed
#         pass

# # model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.95)\


# # def get_embeddings(texts, vectorizer=None):
# #     if isinstance(texts, str):
# #         texts = [texts]
    
# #     # Use vLLM to generate a summary or representation of each text
# #     prompt_template = "Summarize the following code in one sentence: {}"
# #     prompts = [prompt_template.format(text) for text in texts]
    
# #     sampling_params = SamplingParams(temperature=0.0, max_tokens=50)  # Adjust max_tokens as needed
# #     outputs = llm.generate(prompts, sampling_params)
    
# #     # Extract the generated summaries
# #     summaries = [output.outputs[0].text.strip() for output in outputs]
    
# #     # Use TF-IDF to create embeddings from the summaries
# #     if vectorizer is None:
# #         vectorizer = TfidfVectorizer()
# #         embeddings = vectorizer.fit_transform(summaries)
# #     else:
# #         embeddings = vectorizer.transform(summaries)
    
# #     return vectorizer, embeddings.toarray()

# # def rerank_docs(query: str, retrieved_docs: List[dict], top_k: int = 3) -> List[dict]:
# #     # First, get embeddings for documents
# #     vectorizer, docs_embeddings = get_embeddings([doc["code"] for doc in retrieved_docs])
    
# #     # Then, use the same vectorizer for the query
# #     _, query_embeddings = get_embeddings(query, vectorizer)

# #     similarities = cosine_similarity(query_embeddings, docs_embeddings)[0]
# #     docs_df = pd.DataFrame(retrieved_docs)
# #     docs_df["similarity"] = similarities
# #     docs_df = docs_df.sort_values(by="similarity", ascending=False)
# #     docs_df = docs_df.drop_duplicates(subset=["code"], keep="first")
# #     return docs_df.head(top_k).to_dict(orient="records")



# def main():
#     parser = ArgumentParser()
#     parser.add_argument("-c", "--cache-directory", type=Path, default="data/cache")
#     parser.add_argument("--reload-cache", action="store_true")
#     args = parser.parse_args()

#     if not args.cache_directory.exists():
#         args.cache_directory.mkdir(parents=True)

#     else:
#         if (args.cache_directory / "retriever").exists():
#             retriever = Retriever.load(args.cache_directory / "retriever")
#         elif (args.cache_directory / "preprocessed.jsonl").exists():
#             preprocessed_df = preprocess_data(
#                 args.cache_directory / "raw.jsonl",
#                 args.cache_directory / "preprocessed.jsonl",
#                 args.reload_cache,
#             )
#             retriever.docs = Retriever(preprocessed_df.to_dict(orient="records"))
#             retriever.index()
#             retriever.save(args.cache_directory / "retriever")
#         else:
#             raw_df = get_code_contests_data(
#                 args.cache_directory / "raw.jsonl", args.reload_cache
#             )
#             preprocessed_df = preprocess_data(
#                 args.cache_directory / "raw.jsonl",
#                 args.cache_directory / "preprocessed.jsonl",
#                 args.reload_cache,
#             )
            # # model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.95)\


# # def get_embeddings(texts, vectorizer=None):
# #     if isinstance(texts, str):
# #         texts = [texts]
    
# #     # Use vLLM to generate a summary or representation of each text
# #     prompt_template = "Summarize the following code in one sentence: {}"
# #     prompts = [prompt_template.format(text) for text in texts]
    
# #     sampling_params = SamplingParams(temperature=0.0, max_tokens=50)  # Adjust max_tokens as needed
# #     outputs = llm.generate(prompts, sampling_params)
    
# #     # Extract the generated summaries
# #     summaries = [output.outputs[0].text.strip() for output in outputs]
    
# #     # Use TF-IDF to create embeddings from the summaries
# #     if vectorizer is None:
# #         vectorizer = TfidfVectorizer()
# #         embeddings = vectorizer.fit_transform(summaries)
# #     else:
# #         embeddings = vectorizer.transform(summaries)
    
# #     return vectorizer, embeddings.toarray()

# # def rerank_docs(query: str, retrieved_docs: List[dict], top_k: int = 3) -> List[dict]:
# #     # First, get embeddings for documents
# #     vectorizer, docs_embeddings = get_embeddings([doc["code"] for doc in retrieved_docs])
    
# #     # Then, use the same vectorizer for the query
# #     _, query_embeddings = get_embeddings(query, vectorizer)

# #     similarities = cosine_similarity(query_embeddings, docs_embeddings)[0]
# #     docs_df = pd.DataFrame(retrieved_docs)
# #     docs_df["similarity"] = similarities
# #     docs_df = docs_df.sort_values(by="similarity", ascending=False)
# #     docs_df = docs_df.drop_duplicates(subset=["code"], keep="first")
# #     return docs_df.head(top_k).to_dict(orient="records")


#             retriever.docs = Retriever(preprocessed_df.to_dict(orient="records"))
#             retriever.index()
#             query = """
#             def fibonacci(n):
#                 if n <= 1:
#                     return n
#                 else:
#                     return fibonacci(n-1) + fibonacci(n-2)
#             """
#             results = retriever.retrieve(query, k=10)
            
#             if results:
#                 print("Initial results:")
#                 for i, doc in enumerate(results[:5], 1):
#                     print(f"Result {i}:")
#                     print(f"Description: {doc['description'][:100]}...")
#                     print(f"Code: {doc['code'][:100]}...")
#                     print()
                
#                 # Rerank results
#                 # reranked_results = rerank_docs(query, results)
                
#                 # print("Reranked results:")
#                 # for i, doc in enumerate(reranked_results, 1):
#                 #     print(f"Result {i}:")
#                 #     print(f"Description: {doc['description'][:100]}...")
#                 #     print(f"Code: {doc['code'][:100]}...")
#                 #     print(f"Similarity: {doc['similarity']:.4f}")
#                 #     print()
#             else:
#                 print("No results found.")
            
#             # Print normalized query for debugging
#             print("Normalized Query:")
#             print(normalize_code(query))
#             retriever.save(args.cache_directory / "retriever")
# if __name__ == "__main__":
#         main()