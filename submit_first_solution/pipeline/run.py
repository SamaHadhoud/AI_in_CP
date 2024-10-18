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
from one_shot import *
from mini_lib.reflection_logic import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from models import get_vllm, get_embedding_model
from models.vllm_model import model_name
import time
from retrieval_logic import *

import asyncio

@dataclass
class Args(simple_parsing.Serializable):
    problem_names: List[str] = field(default_factory=lambda:  [

        "line_by_line","walk_the_line", "fall_in_line", "line_of_delivery_1", "line_of_delivery_2"])
        
        # "cheeseburger_corollary_ch1", 
        # "cheeseburger_corollary_ch2", "dim_sum_delivery", "two_apples_a_day", "road_to_nutella"])
        # # "here_comes_santa_claus", 
        # "sum_41_ch1",
        # "sum_41_ch2", 
        
        # "back_in_black_ch1", "back_in_black_ch2", "today_is_gonna_be_a_great_day", "bohemian_rap-sody"] ) # list of problems to solve
    # folder_path: Path = Path("./dataset/2023/practice/")
    folder_path: Path = Path("./dataset/contestData_practice2024")
    weave_log: bool = True
    use_images: bool = False
    save_output: bool = True
    debug: bool = False
    timeout: int = 40
    max_attempts: int = 20
    cache_directory: Path = Path("data/cache")

@weave.op()
async def solve_single_problem(args: Args, problem_name: str, retriever):
    """Solve a single problem and log results in Weave."""
    problem = Problem.from_name(problem_name, args.folder_path)
    logging.info(f"Solving problem: {problem_name}")
    initial_draft_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout)

    solution_attempts = []
    
    solution_attempts.append(initial_draft_solution)

    weave.save({f"{problem_name}_initial_draft_attempt": initial_draft_solution})
    logging.info(f"Initial draft attempt - Status: {initial_draft_solution.status}")
    
    # Check full input on initial solution
    logging.info("> Checking full input on initial solution...")
    initial_draft_full_input_result = solve_full_input(problem, initial_draft_solution, args)
    weave.save({f"{problem_name}_initial_full_draft_input_result": initial_draft_full_input_result})

    # Use the generated code as a query for retrieval
    # Retrieve documents
    print("\nAttempting to retrieve documents...")
    retrieved_docs = retriever.retrieve(initial_draft_solution.code, k=500)

    if not retrieved_docs:
        print("No documents retrieved. Check the query processing.")
    else:
        print("\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\nDocument {i}:")
            print(f"Code:\n{doc['cleaned_code'][:200]}...")  # Print first 200 characters of the code
            print(f"Description: {doc['description'][:100]}...")  # Print first 100 characters of the description

    reranked_docs = rerank_docs(problem, initial_draft_solution.code, retrieved_docs, top_k=3)

    if not reranked_docs:
        print("No documents after reranking. Check the reranking process.")
    else:
        print("\nReranked Documents:")
        for i, doc in enumerate(reranked_docs, 1):
            # Remove 'normalized_code' from each document
            doc.pop('normalized_code', None)
            
            print(f"\nDocument {i}:")
            print(f"Description:\n{doc['description']}")
            print(f"\nCode:\n{doc['cleaned_code']}")
            print(f"Similarity: {doc['similarity']:.4f}")


    # Prepare examples for the prompt
    examples = format_examples(reranked_docs)
    
    initial_solution = solve_problem(problem, use_images=args.use_images, timeout=args.timeout, examples = examples)
    
    # best_initial_solution_list = rank_solutions([initial_solution, initial_draft_solution])
    # best_initial_solution = best_initial_solution_list[0]
    
    solution_attempts.append(initial_solution)

    weave.save({f"{problem_name}_initial_attempt": initial_solution})
    logging.info(f"Initial attempt - Status: {initial_solution.status}")
    
    # Check full input on initial solution
    logging.info("> Checking full input on initial solution...")
    initial_full_input_result = solve_full_input(problem, initial_solution, args)
    weave.save({f"{problem_name}_initial_full_input_result": initial_full_input_result})
    
    if not (initial_solution.status == 'success' and initial_solution.test_cases['matches']):
        current_solution = initial_solution
        for attempt in range(args.max_attempts):
            
            logging.info(f"Attempt {attempt + 1} - Reflecting and improving...")
            reflection_result = await reflection(problem.as_xml, current_solution, examples)
            improved_solution = await improve_solution(problem.as_xml, current_solution,reflection_result, examples)
            
            solution_result = try_solution(problem, improved_solution, args.timeout)
            solution_attempts.append(solution_result)
            current_solution = solution_result

            weave.save({f"{problem_name}_attempt_{attempt + 1}": solution_result})
            logging.info(f"Attempt {attempt + 1} - Status: {solution_result.status}")
            
            if solution_result.status == 'success' and solution_result.test_cases['matches']:
                break

    ranked_solutions = rank_solutions(solution_attempts)
    best_solution = ranked_solutions[0]
    
    weave.save({f"{problem_name}_best_solution": best_solution})
    logging.info(f"Best solution status: {best_solution.status}")

    best_solution_result = try_solution(problem, best_solution.code, args.timeout)
    # Check full input on best solution
    logging.info("> Checking full input on best solution...")
    best_full_input_result = solve_full_input(problem, best_solution, args)
    weave.save({f"{problem_name}_best_full_input_result": best_full_input_result})

    return {
        "initial_solution": initial_solution,
        "initial_full_input_result": initial_full_input_result,
        "best_solution": best_solution,
        "best_full_input_result": best_full_input_result
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
    input=problem.get_input()
    if not input:
        problem.save_code(solution.code)
        result["output_saved"] = True
        return "no input"
    
    expected_output = problem.get_output()
    try:
        start_time = time.time()
        generated_output = run(solution.code, input=problem.get_input(), timeout=args.timeout)
        execution_time = time.time() - start_time
        matches = check_solution(expected_output, generated_output)
        result = {
            "status": "success",
            "matches": matches,
            "execution_time": execution_time
        }
        
        if args.save_output:
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
            matches = check_solution(expected_output, generated_output)
            result = {
                "status": "success_with_extended_timeout",
                "matches": matches,
                "execution_time": execution_time
            }
            
            if args.save_output:
                problem.save_code(solution.code)
                result["output_saved"] = True
        except TimeoutException:
            result = {
                "status": "timeout",
                "error": "The solution took too long to execute even with extended timeout."
            }

            if args.save_output:
                problem.save_code(solution.code)
                result["output_saved"] = True

    except Exception as e:
        result = {"status": "error", "error": str(e)}
    
    return result




@weave.op()
async def main(args: Args):
    """Main function to solve multiple problems sequentially."""
    setup_logger(args.debug)
    if not args.cache_directory.exists():
        args.cache_directory.mkdir(parents=True)

    if args.weave_log:
        weave.init(f"hack-cup-{model_name.replace('/', '-')}")

    retriever = Retriever("AlaaAhmed2444/rag_full")
    



    all_results = {}
    for problem_name in args.problem_names:
        problem_results = await solve_single_problem(args, problem_name, retriever)
        all_results[problem_name] = problem_results
        
        logging.info(f"Results for {problem_name}:")
        logging.info(f"Initial solution status: {problem_results['initial_solution'].status}")
        logging.info(f"Initial full input result: {problem_results['initial_full_input_result']}")
        logging.info(f"Best solution status: {problem_results['best_solution'].status}")
        logging.info(f"Best full input result: {problem_results['best_full_input_result']}")
        logging.info("---")

    weave.save(all_results, "all_problems_results")

if __name__ == "__main__":
    args = simple_parsing.parse(Args)
    asyncio.run(main(args))