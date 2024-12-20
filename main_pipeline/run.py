from dataclasses import dataclass
from pathlib import Path
import logging
from dataclasses import dataclass, field
import weave
import simple_parsing
from vllm import LLM, SamplingParams
from mini_lib.problem24 import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run,TimeoutException
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from zero_shot import *
from reflection_logic import *
from choose_best import *
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
        #practice 2024
        "Walk the Line", 
        "Line by Line",
        "Fall in Line", 
        "Line of Delivery (Part 1)", "Line of Delivery (Part 2)"
        #round1 2024
        # "Prime Subtractorization", 
        # "Subsonic Subway"
        # "Substantial Losses", 
        # "Substitution Cipher", 
        # "Wildcard Submissions"     
        ])
        
    problem_letters: List[str] = field(default_factory=lambda:  ["a", "b", "c", "d", "d"])
    problem_round: str= "practice"
    folder_path: Path = Path("./2024-practice")
    weave_log: bool = True
    use_images: bool = False
    save_output: bool = True
    debug: bool = False
    timeout: int = 30
    max_attempts: int = 20
    retrive_flag: bool = False
    few_shot_cot_examples_flag: bool = True
    choose_best_flag: bool = False
    heurstic_compare:bool = True
    cache_directory: Path = Path("data/cache")

async def get_few_shot_cot_examples(problem_letter, problem_round):
    filename = f"./examples/{problem_round}_{problem_letter}.txt"
    with open(filename, 'r') as f:
        content = f.read()
    return content
@weave.op()
async def solve_single_problem(args: Args, problem_name: str, problem_letter, retriever):
    """Solve a single problem and log results in Weave."""
    examples = ""
    if(args.few_shot_cot_examples_flag):
        examples = await get_few_shot_cot_examples(problem_letter, args.problem_round)
    

    problem = Problem.from_name(problem_name, args.folder_path)
    logging.info(f"Solving problem: {problem_name}")

    analysis = self_reflection_on_problem(problem.as_xml)
    initial_draft_solution = solve_problem(problem, analysis, use_images=args.use_images, timeout=args.timeout, examples = examples)

    solution_attempts = []
    
    solution_attempts.append(initial_draft_solution)

    weave.save({f"{problem_name}_initial_draft_attempt": initial_draft_solution})
    logging.info(f"Initial draft attempt - Status: {initial_draft_solution.status}")
    
    # Check full input on initial solution
    logging.info("> Checking full input on initial solution...")
    initial_draft_full_input_result = solve_full_input(problem, initial_draft_solution, args)
    weave.save({f"{problem_name}_initial_full_draft_input_result": initial_draft_full_input_result})
    
    if(args.choose_best_flag):
        initial_draft_solution = solve_problem_choose_best(problem, analysis, use_images=args.use_images, timeout=args.timeout, examples=examples, heurstic_compare = args.heurstic_compare)

        solution_attempts = []
        
        solution_attempts.append(initial_draft_solution)

        weave.save({f"{problem_name}_initial_draft_choose_best_attempt": initial_draft_solution})
        logging.info(f"Initial draft choose_best attempt - Status: {initial_draft_solution.status}")
        
        # Check full input on initial solution
        logging.info("> Checking full input on initial choose_best solution...")
        initial_draft_full_input_result = solve_full_input(problem, initial_draft_solution, args)
        weave.save({f"{problem_name}_initial_full_draft_choose_best__input_result": initial_draft_full_input_result})
    
    
    if  args.retrive_flag:
        #TO-DO: change retrival logic 
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
    
        initial_solution = solve_problem(problem, analysis, use_images=args.use_images, timeout=args.timeout, examples = examples)
        
        solution_attempts.append(initial_solution)

        weave.save({f"{problem_name}_initial_attempt": initial_solution})
        logging.info(f"Initial attempt - Status: {initial_solution.status}")
        
        # Check full input on initial solution
        logging.info("> Checking full input on initial solution...")
        initial_full_input_result = solve_full_input(problem, initial_solution, args)
        weave.save({f"{problem_name}_initial_full_input_result": initial_full_input_result})
    else:
        initial_solution = initial_draft_solution
        initial_full_input_result = initial_draft_full_input_result
    if not (initial_solution.status == 'success' and initial_solution.test_cases['matches']):
        current_solution = initial_solution
        for attempt in range(args.max_attempts):
            
            logging.info(f"Attempt {attempt + 1} - Reflecting and improving...")
            reflection_result = await reflection(problem.as_xml, analysis,  current_solution, examples)
            improved_solution = await improve_solution(problem.as_xml, analysis, current_solution,reflection_result, examples)
            
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

    x = "choose_best-" if args.choose_best_flag else ""
    if args.choose_best_flag and args.heurstic_compare:
        x+="heurstic_compare-"
    if args.few_shot_cot_examples_flag:
        x+="few_shot_cit_examples"
    if args.weave_log:
        weave.init(f"with-analysis-hack-cup-{x}{model_name.replace('/', '-')}")

    retriever = None
    if(args.retrive_flag):
        retriever = Retriever("AlaaAhmed2444/rag_full")
    

    all_results = {}
    for idx, problem_name in enumerate(args.problem_names):
        problem_results = await solve_single_problem(args, problem_name, args.problem_letters[idx], retriever)
        all_results[problem_name] = problem_results

    weave.save(all_results, "all_problems_results")

if __name__ == "__main__":
    args = simple_parsing.parse(Args)
    asyncio.run(main(args))