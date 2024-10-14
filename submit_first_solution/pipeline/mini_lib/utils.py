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

import multiprocessing
import time
import logging
import traceback
import sys
from typing import Optional

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
    i = 1
    for expected_line, actual_line in zip(expected_lines, actual_lines):
        expected_line = expected_line.strip()
        actual_line = actual_line.strip()
        
        if expected_line == actual_line:
            matches += 1  # +1 for the whole line match
        else:
            offending_cases.append([i, expected_line, actual_line])
        i+=1
    return {"actual" : actual, "matches": matches == len(expected_lines), "total": len(expected_lines), "len_offending_cases": len(offending_cases), "len_passed_cases": len(expected_lines) - len(offending_cases),"offending_cases": offending_cases}

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

def worker(code: str, input: Optional[str], result_queue):
    try:
        # Create a dictionary to serve as a global namespace
        global_namespace = {}
        
        # Execute the code in the global namespace
        exec(code, global_namespace)
        
        # Call the solve function with the input
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve'](input)
            result_queue.put(result)
        else:
            raise Exception("No 'solve' function found in the code")
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        _, _, tb = sys.exc_info()
        
        # Extract the relevant part of the traceback
        tb_list = traceback.extract_tb(tb)
        relevant_tb = None
        for frame in reversed(tb_list):
            if frame.filename == '<string>':
                relevant_tb = frame
                break
        
        if relevant_tb:
            line_no = relevant_tb.lineno
            # If the line is not available in the traceback, extract it from the code
            if not relevant_tb.line:
                code_lines = code.split('\n')
                if 0 <= line_no - 1 < len(code_lines):
                    error_line = code_lines[line_no - 1].strip()
                else:
                    error_line = "Unable to retrieve the line"
            else:
                error_line = relevant_tb.line.strip()
        else:
            line_no = "unknown"
            error_line = "Unable to retrieve the line"
        
        error_info = {
            'type': error_type,
            'message': error_message,
            'line_no': line_no,
            'error_line': error_line
        }
        result_queue.put(('error', error_info))
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
        if isinstance(result, tuple) and result[0] == 'error':
            error_info = result[1]
            error_message = f"Error: {error_info['type']}: {error_info['message']}\nLine {error_info['line_no']}: {error_info['error_line']}"
            raise Exception(error_message)
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
        logging.error(f"Error executing code: {str(e)}")
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
