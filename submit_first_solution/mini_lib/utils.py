import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
import time
from typing import Optional, List

from rich.logging import RichHandler
import subprocess
import tempfile
import os
tempfile.tempdir = "./dataset//2023//practice"
import weave

def load_jsonl(file: Path) -> List[dict]:
    """Load a JSONL file"""
    with open(file, 'r') as f:
        return [json.loads(line) for line in f]

class TimeoutException(Exception):
    pass

def setup_logger(debug = False, silence_openai = True):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    # silence openai logger
    if silence_openai:
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

import re

def maybe_remove_backticks(solution: str) -> str:
    "Remove backticks from the solution"
    solution = solution.strip()
    solution = re.sub(r'^```python\s*', '', solution)
    solution = re.sub(r'\s*```$', '', solution)
    return solution

def maybe_remove_backticks_cpp(solution: str) -> str:
    "Remove backticks from the solution"
    solution = solution.strip()
    solution = re.sub(r'^```cpp\s*', '', solution)
    solution = re.sub(r'\s*```$', '', solution)
    return solution

@weave.op
def check_solution(expected: str, actual: str) -> dict:
    "Check the solution against the expected output"
    matches = 0
    expected_lines = expected.strip().split("\n")
    logging.debug(f"Expected lines: {expected_lines}")
    actual_lines = actual.strip().split("\n")
    logging.debug(f"Actual lines: {actual_lines}")
    offending_cases = []
    for expected_line, actual_line in zip(expected_lines, actual_lines):
        expected_line = expected_line.strip()
        actual_line = actual_line.strip()
        
        if expected_line == actual_line:
            matches += 1  # +1 for the whole line match
        else:
            offending_cases.append((expected_line, actual_line))
    return {"matches": matches == len(expected_lines), "total": len(expected_lines), "len_offending_cases": len(offending_cases), "len_passed_cases": len(expected_lines) - len(offending_cases),"offending_cases": offending_cases}

class TimeoutException(Exception):
    pass

# def run_with_timeout(code: str, input: Optional[str], timeout: int):
#     vars = {}
#     try:
#         exec(code, vars)
#     except Exception as e:
#         logging.error(f"The generated code is not valid: {code}")
#         raise e

#     fn = vars.get("solve", lambda x: x)
#     return fn(input)

import multiprocessing
import time
import logging

class TimeoutException(Exception):
    pass

def worker(code, input, result_queue):
    try:
        vars = {}
        exec(code, vars)
        fn = vars.get("solve", lambda x: x)
        result = fn(input)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)

def run_with_timeout(code: str, input: Optional[str], timeout: int):
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=(code, input, result_queue))
    process.start()

    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutException(f"Function call timed out after {timeout} seconds")

    if not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    else:
        raise Exception("No result produced")

def run(code: Optional[str] = None, input: Optional[str] = None, timeout: int = 60):
    logging.info("Running solution synchronously...")
    t0 = time.perf_counter()
    try:
        result = run_with_timeout(code, input, timeout)
        logging.info("Execution completed successfully.")
        return result
    except TimeoutException as e:
        logging.error(f"Function call timed out after {timeout} seconds")
        raise e
    except Exception as e:
        logging.error(f"Error executing code: {e}")
        raise e
    finally:
        t1 = time.perf_counter()
        logging.info(f"Code solution runtime: {t1 - t0:.2f} seconds")

# We can keep the async version if needed, but it might not be necessary anymore
async def arun(code: Optional[str] = None, input: Optional[str] = None, timeout: int = 60):
    logging.info(f"Running solution asynchronously with {timeout}s timeout...")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run, code, input, timeout)



def run_cpp(code: str, input: str, timeout: int = 10) -> str:
    tmpdir =".//dataset//2023//practice"
    cpp_file = os.path.join(tmpdir, "solution.cpp")
    exe_file = os.path.join(tmpdir, "solution")
    
    # Write the C++ code to a file
    try:
        with open(cpp_file, "w") as f:
            f.write(code)
        logging.info(f"C++ code written to {cpp_file}")
    except IOError as e:
        logging.error(f"Error writing C++ code to file: {e}")
        return f"Error writing C++ code to file: {e}"
    
    # Compile the C++ code
    try:
        compile_process = subprocess.run(["g++", "-std=c++11", "-O3", cpp_file, "-o", exe_file], capture_output=True, text=True, check=True)
        logging.info("C++ code compiled successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Compilation error: {e.stderr}")
        return f"Compilation error:\n{e.stderr}"
    
    # Run the compiled executable
    try:
        process = subprocess.run([exe_file], input=input, capture_output=True, timeout=timeout)
        logging.info("Executable ran successfully")
        return process.stdout.decode('utf-8')
    except subprocess.TimeoutExpired:
        logging.error("Execution timed out")
        return "Timeout"
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Test check_solution
    expected = "Case #1: YES\nCase #2: NO\nCase #3: YES"
    actual = "Case #1: YES\nCase #2: Yes\nCase #3: YES"
    result = check_solution(expected, actual)
    assert result["matches"] == 2, "Expected 2 matches"
    assert result["total"] == 3, "Expected 3 total lines"
    assert len(result["offending_cases"]) == 1, "Expected 1 offending case"
    assert result["offending_cases"][0] == ("Case #2: NO", "Case #2: Yes"), "Unexpected offending case"

    # Test maybe_remove_backticks
    assert maybe_remove_backticks("print('hello')\n```") == "print('hello')"
    assert maybe_remove_backticks("print('hello')\n```  ") == "print('hello')"
    assert maybe_remove_backticks("```python\nprint('hello')") == "print('hello')"
    assert maybe_remove_backticks("```python\nprint('hello')\n```") == "print('hello')"

    # test exec
    code = "def solve(x: int):\n    return x + 1"
    input = 2
    result = run(code, input)
    assert result == 3, "Expected 3"

    # async test
    result = asyncio.run(arun(code, input))
    assert result == 3, "Expected 3"
    print("All tests passed!")

