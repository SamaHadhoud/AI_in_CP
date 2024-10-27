
from pydantic import BaseModel
from pydantic.fields import Field
from typing import List, Optional
from model import call_model, count_tokens
from mini_lib.problem24 import Problem
from mini_lib.utils import maybe_remove_backticks, check_solution, setup_logger, run,TimeoutException
import re, time
import weave
import logging
from zero_shot import system_prompt, SolutionAttempt, system_prompt_with_examples
import yaml

class Solution(BaseModel):
    name: str
    problem_variables: str
    approach_summary: str
    detailed_solution: str
    mathematical_foundation: str
    correctness_proof: str
    complexity_analysis: str
    implementation_notes: str
    labels: List[str]


class ProblemSolutions(BaseModel):
    possible_solutions: List[Solution] = Field(max_items=3)


class SolutionParser:
    """Parser for solution responses in YAML format"""

    @staticmethod
    def clean_yaml_content(content: str) -> str:
            # Remove markdown code block markers
        content = re.sub(r'```yaml\s*|```\s*', '', content)
        
        # Find the start of YAML content
        yaml_start = content.find('solutions:')
        if yaml_start == -1:
            raise ValueError("No solutions section found")
        content = content[yaml_start:]
        
        # Split into lines and process each line
        lines = content.split('\n')
        cleaned_lines = []
        in_labels = False
        # Track YAML structure to know when it ends
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if not stripped:
                continue
                
            # Check if we've reached the end of the YAML structure
            if (i > 0 and 
                not stripped.startswith(' ') and 
                not stripped.startswith('-') and 
                not stripped.startswith('solutions:')):
                break
                
            #  Check if we're entering labels section
            if stripped.strip() == 'labels:':
                in_labels = True
                cleaned_lines.append(stripped)
                continue
                
            # Handle labels section
            if in_labels:
                if not stripped.startswith(' '):
                    in_labels = False
                else:
                    # Convert "[category]: value" format to simple "value"
                    match = re.search(r'-\s*\[(.*?)\]:\s*(.*)', stripped)
                    if match:
                        indent = len(re.match(r'^\s*', stripped).group())
                        cleaned_lines.append(' ' * indent + '- ' + match.group(2).strip())
                        continue
            
            cleaned_lines.append(stripped)
            
        return '\n'.join(cleaned_lines)
    
    
    @staticmethod
    def parse_solutions(response: str) -> List[Solution]:
        try:
            yaml_content = SolutionParser.clean_yaml_content(response)
            parsed = yaml.safe_load(yaml_content)
            
            if not parsed or 'solutions' not in parsed:
                raise ValueError("Invalid YAML structure")
                
            solutions = []
            for sol in parsed['solutions']:
                if 'solution' not in sol:
                    continue
                    
                sol_data = sol['solution']
                # Updated required fields to match new YAML structure
                required_fields = [
                    'name', 
                    'problem_variables',
                    'approach_summary',
                    'detailed_solution',
                    'mathematical_foundation',
                    'correctness_proof',
                    'complexity_analysis',
                    'implementation_notes',
                    'labels'
                ]
                
                if not all(field in sol_data for field in required_fields):
                    continue
                    
                # Clean and validate the data
                labels = [label.strip() for label in sol_data['labels'] if isinstance(label, str)]
                
                solution = Solution(
                    name=sol_data['name'].strip(),
                    problem_variables=sol_data['problem_variables'].strip(),
                    approach_summary=sol_data['approach_summary'].strip(),
                    detailed_solution=sol_data['detailed_solution'].strip(),
                    mathematical_foundation=sol_data['mathematical_foundation'].strip(),
                    correctness_proof=sol_data['correctness_proof'].strip(),
                    complexity_analysis=sol_data['complexity_analysis'].strip(),
                    implementation_notes=sol_data['implementation_notes'].strip(),
                    labels=labels
                )
                solutions.append(solution)
            
            return solutions if solutions else [SolutionParser._create_fallback_solution()]
            
        except Exception as e:
            logging.error(f"Error parsing solutions: {str(e)}")
            return [SolutionParser._create_fallback_solution()]

    @staticmethod
    def _create_fallback_solution() -> Solution:
        return Solution(
            name="Basic Implementation",
            problem_variables="Standard input variables used",
            approach_summary="Direct implementation approach",
            detailed_solution="Basic implementation following problem requirements",
            mathematical_foundation="Basic mathematical operations",
            correctness_proof="Follows problem specifications directly",
            complexity_analysis="Time: O(n), Space: O(1)",
            implementation_notes="Simple implementation strategy",
            labels=["implementation"]
        )
    
@weave.op
def get_all_possible_solutions(problem: Problem, analysis, examples):
    system_prompt = """
You are a world-class competitive programmer and mathematical theorist analyzing code contest problems.
Your expertise spans algorithms, data structures, and mathematical optimization.

You have previously solved the following problems in this competition:
<examples>
{examples}
</examples>

CORE RESPONSIBILITIES:
Generate ALL viable solutions considering different paradigms like:
□ Brute force approaches with optimizations
□ Dynamic programming solutions
□ Greedy algorithm approaches
□ Divide and conquer strategies
□ Mathematical/formula-based solutions
□ Space-time trade-off variants
□ Different data structure choices
□ Algorithm-specific optimizations

For EACH possible approach, ensure:
1. CORRECTNESS & COMPLETENESS:
□ Full coverage of problem requirements
□ Handling of all constraints
□ Generalization beyond test cases
□ Edge case management
□ Input validation strategy

2. MATHEMATICAL RIGOR:
□ Formal mathematical notation
□ Complete proofs of correctness
□ Derivation of formulas
□ Error bound analysis
□ Numerical stability proof

3. ALGORITHMIC EFFICIENCY:
□ Time complexity analysis
□ Space complexity bounds
□ Optimization opportunities
□ Performance bottlenecks
□ Resource utilization

4. IMPLEMENTATION FEASIBILITY:
□ Practical coding considerations
□ Memory management
□ Error handling
□ Input/output processing
□ Code organization

YOUR RESPONSE MUST FOLLOW THIS YAML FORMAT FOR EACH POSSIBLE SOLUTION:

solutions:
- solution:
    name: "[Distinct algorithmic approach name]"
    problem_variables: |
        [Original variable definitions]
        - List all variables from problem statement
        - Maintain exact naming and notation
        - Document relationships between variables
    
    approach_summary: |
        [High-level solution strategy]
        1. Core algorithm selection rationale
        2. Key data structure choices
        3. Mathematical foundation
        4. Optimization approach
    
    detailed_solution: |
        [Step-by-step implementation]
        1. Initialization
        - Input processing
        - Data structure setup
        - Variable initialization
        
        2. Core Algorithm
        - Main processing steps
        - Key transformations
        - Optimization techniques
        
        3. Output Generation
        - Result compilation
        - Format conversion
        - Validation steps
    
    mathematical_foundation: |
        [Complete mathematical framework]
        1. Formal Definitions:
        - Variables and domains
        - Constraints and invariants
        - Objective functions
        
        2. Key Formulas:
        - Formula 1: [LaTeX/plaintext]
            ↳ Derivation
            ↳ Proof of correctness
            ↳ Application in solution
        
        3. Numerical Analysis:
        - Stability proof
        - Error bounds
        - Precision requirements
    
    correctness_proof: |
        [Formal verification]
        1. Invariants:
        - Loop invariants
        - Data structure invariants
        - State invariants
        
        2. Termination:
        - Convergence proof
        - Bound on iterations
        - Resource limits
        
        3. Edge Cases:
        - Boundary conditions
        - Special inputs
        - Error conditions
    
    complexity_analysis: |
        [Detailed resource analysis]
        1. Time Complexity:
        - Overall: O(X)
        - Breakdown by component
        - Worst-case scenarios
        - Amortized analysis
        
        2. Space Complexity:
        - Overall: O(Y)
        - Auxiliary space
        - Stack space
        - Cache considerations
        
        3. Numerical Complexity:
        - Precision requirements
        - Error propagation
        - Stability analysis
    
    implementation_notes: |
        [Practical considerations]
        1. Code Structure
        2. Critical Optimizations
        3. Error Handling
        4. Testing Strategy
    
    labels:
        - [algorithm_category]
        - [data_structures]
        - [mathematical_concepts]
        - [optimization_techniques]

- solution:
    name: "[Next distinct approach]"
    [... Same structure for each possible solution ...]
"""

    user_prompt = """
PROBLEM ANALYSIS REQUEST

Given Problem:
{problem.as_xml}


Generate ALL possible comprehensive solutions following the specified YAML format.
Consider every viable approach, ensuring each is distinct and valuable.
"""

    messages = [
        {"role": "system", "content": system_prompt.format(examples = examples)},
        {"role": "user", "content": user_prompt.format(problem=problem)}
    ]
    parser = SolutionParser()
    try:
        response = call_model(messages=messages)
        solutions = parser.parse_solutions(response)
        
        logging.info(f"Successfully generated {len(solutions)} solutions")
        return solutions
        
    except Exception as e:
        logging.error(f"Error in get_all_possible_solutions: {str(e)}")
        return [parser._create_fallback_solution()]


@weave.op
def select_best_solution(solutions: str, problem: Problem, analysis: str, examples) -> Solution:
    # Format solutions list nicely
    solutions_text = "\n\n".join([
        f"Solution {i+1}:\n"
        f"Name: {sol.name}\n"
        f"Problem Variables: {sol.problem_variables}\n"
        f"Approach Summary: {sol.approach_summary}\n"
        f"Detailed Solution: {sol.detailed_solution}\n"
        f"Mathematical Foundation: {sol.mathematical_foundation}\n"
        f"Correctness Proof: {sol.correctness_proof}\n"
        f"Complexity Analysis: {sol.complexity_analysis}\n"
        f"Implementation Notes: {sol.implementation_notes}\n"
        f"Labels: {', '.join(sol.labels)}\n"
        for i, sol in enumerate(solutions)
    ])

    system_prompt = """
    You are a senior competitive programming judge. 

    You have previously solved the following problems in this competition:
    <examples>
    {examples}
    </examples>
        
    Given multiple possible solutions to a problem, your task is to select the best solution based on the following criteria:

    1. Correctness: 
       - Mathematical correctness and proof validity
       - Numerical stability and accuracy where applicable
    2. Mathematical Efficiency:
       - Optimality of mathematical approach
       - Elegance of mathematical solution
    3. Implementation Efficiency:
       - Time and space complexity with mathematical justification
       - Practical performance considerations
    4. Robustness:
       - Numerical robustness for mathematical operations
       - Handling of edge cases and special values
    5. Clarity:
       - Clear explanation of mathematical concepts
       - Well-documented derivations and proofs

    Remember: Evaluate both theoretical mathematical correctness and practical implementation aspects.
    """

    user_prompt = """
    Problem Statement:
    {problem.as_xml}

    Available Solutions:
    {solutions_text}

    Please analyze these solutions and select the best one. Explain your choice in YAML format:

    selected_solution:
      name: <solution_name>
      mathematical_analysis: |
        - Mathematical concepts utilized
        - Correctness of proofs and derivations
        - Numerical considerations
      rationale: <explanation of why this is the best choice>
      key_advantages:
        - <advantage 1>
        - <advantage 2>
      potential_risks:
        - <risk 1>
        - <risk 2>
    """

    messages = [
        {"role": "system", "content": system_prompt.format(examples=examples)},
        {"role": "user", "content": user_prompt.format(
            problem=problem,
            # analysis=analysis,
            solutions_text=solutions_text
        )}
    ]
    try:
        response = call_model(messages=messages)
        
        # Extract selected solution name
        selected_match = re.search(r'name:\s*(.*?)\n', response)
        if selected_match:
            selected_name = selected_match.group(1).strip()
            # Find the matching solution
            for solution in solutions:
                if selected_name.lower().strip() in solution.name.lower().strip():
                    return solution
                    
        # If no match found or parsing failed, select first solution
        logging.warning("Could not parse selected solution, returning first solution")
        return solutions[0]
        
    except Exception as e:
        logging.error(f"Error selecting best solution: {str(e)}")
        return solutions[0]



@weave.op
def generate_code_from_best_solution(
    problem: Problem, 
    analysis: str,
    selected_solution: Solution,
    system_prompt: str, 
    max_attempts: int = 3, 
    use_images: bool = False,
    examples: str = "") -> str:

    if examples:
        system_prompt=system_prompt_with_examples.format(examples=examples)
    
    logging.info(f"Generating code solution for: {problem.name} using solution: {selected_solution.name}")

    # Enhanced system prompt with mathematical details
    enhanced_system_prompt = f"""
{system_prompt}

You will implement the following selected solution:
Solution Name: {selected_solution.name}

Problem Variables:
{selected_solution.problem_variables}

Approach Summary:
{selected_solution.approach_summary}

Detailed Implementation:
{selected_solution.detailed_solution}

Mathematical Foundation:
{selected_solution.mathematical_foundation}

Correctness Proof:
{selected_solution.correctness_proof}

Complexity Analysis:
{selected_solution.complexity_analysis}

Implementation Notes:
{selected_solution.implementation_notes}

Key Implementation Points:
- Focus on these algorithmic concepts: {', '.join(selected_solution.labels)}
"""

    implementation_prompt = """
Please implement the selected solution described above. Your implementation should:
1. Follow the specified approach exactly
2. Meet the stated time and space complexity requirements
3. Include necessary error handling and input validation
4. Be well-commented to explain the implementation
5. Follow the required function signature: def solve(input_data: str) -> str:
6. Include mathematical formula implementations with:
   - Clear documentation of mathematical steps
   - Proper handling of numerical precision
   - Implementation of all required mathematical functions
   - Comments explaining mathematical transformations
7. Include mathematical helper functions with:
   - Documentation of input/output requirements
   - Proper error handling for mathematical edge cases
   - Numerical stability considerations

Remember to consider all edge cases and ensure your implementation is robust and efficient.
"""
    
    for attempt in range(max_attempts):
        logging.info(f"Generating code solution (Attempt {attempt + 1})")

        # Count tokens for problem components
        problem_description_tokens = count_tokens(problem.problem_description)
        sample_input_tokens = count_tokens(problem.sample_input)
        sample_output_tokens = count_tokens(problem.sample_output)
        total_problem_tokens = problem_description_tokens + sample_input_tokens + sample_output_tokens

        # Prepare the full prompt with mathematical requirements
        formatted_prompt = f"""
Problem Statement:
{problem.as_xml}

Problem Analysis:
{analysis}

{implementation_prompt}

Please provide only the Python code, enclosed in triple backticks, like this:

```python
# Your imports here
# Mathematical helper functions here

def solve(input_data: str) -> str:
    # Your code here with mathematical implementations
"""
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": formatted_prompt}
            ] + ([{"type": "image_url", "image_url": {"url": img}} for img in problem.images] if use_images else [])}
        ]

        # Generate initial code
        out = call_model(messages=messages)

        # Extract code prompt with mathematical verification
        extract_prompt = """
    Extract the complete Python code from the previous response. The code should:
    - Be enclosed in triple backticks with the Python language specifier
    - Include all necessary imports at the top
    - Include all mathematical helper functions
    - Contain proper mathematical formula implementations
    - Contain a 'solve' function with the signature: def solve(input_data: str) -> str:
    - Be a complete, runnable Python program that implements the selected solution approach
    - Include necessary comments explaining both algorithmic and mathematical implementations

    Provide only the code, without any additional explanations."""
        
        # Extract clean code
        messages = [
            {"role": "assistant", "content": out},
            {"role": "user", "content": extract_prompt}
        ]

        solution = call_model(messages=messages)
        
        # Extract code between triple backticks
        code_match = re.search(r'```python\n(.*?)```', solution, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            if extracted_code:
                # Verify the code matches the selected solution's complexity and mathematical requirements
                verification_prompt = f"""Verify that this implementation matches the selected solution's requirements:

                Problem Variables:
                {selected_solution.problem_variables}

                Approach Summary:
                {selected_solution.approach_summary}

                Detailed Implementation:
                {selected_solution.detailed_solution}

                Mathematical Foundation:
                {selected_solution.mathematical_foundation}

                Correctness Proof:
                {selected_solution.correctness_proof}

                Complexity Analysis:
                {selected_solution.complexity_analysis}

                Implementation Notes:
                {selected_solution.implementation_notes}

                Code to verify:
                {extracted_code}

                Verify:
                1. Complexity requirements are met
                2. All mathematical formulas are correctly implemented
                3. Mathematical helper functions are properly defined
                4. Numerical stability is considered
                
                Respond with 'VERIFIED' if the implementation matches, or explain what needs to be fixed.
                """
                verification_messages = [
                    {"role": "user", "content": verification_prompt}
                ]
                verification = call_model(messages=verification_messages)
                if "VERIFIED" in verification:
                    return extracted_code
                else:
                    logging.warning(f"Implementation verification failed: {verification}")
                    return extracted_code
            else:
                logging.error("Extracted code is empty")
        else:
            logging.error("No Python code found in the solution")
            
        if attempt < max_attempts - 1:
            logging.warning(f"Attempt {attempt + 1} failed. Retrying...")
        
    logging.error(f"Failed to generate valid code after {max_attempts} attempts")
    return "# Failed to generate valid code"





@weave.op
def solve_problem_choose_best(problem: Problem, analysis, use_images=False, timeout=60, examples="") -> dict:
    solutions = get_all_possible_solutions(problem, analysis, examples)

    # Select best solution
    best_solution = select_best_solution(solutions, problem, analysis, examples)

    code = generate_code_from_best_solution(
        problem, 
        analysis,
        selected_solution=best_solution,
        system_prompt=system_prompt, 
        use_images=use_images,
        examples=examples)
    print("**************************************************************")
    print(code)

    input_data, output = problem.sample_input, problem.sample_output
    try:
        start_time = time.time()
        generated_output = run(code, input=input_data, timeout=timeout)
        execution_time = time.time() - start_time
        test_cases = check_solution(output, generated_output)
        return SolutionAttempt(code=code, status="success", test_cases=test_cases, execution_time=execution_time)
    except TimeoutException:
        return SolutionAttempt(code=code, status="timeout", error= "Execution time limit exceeded")
    except Exception as e:
        return SolutionAttempt(code=code, status="runtime_error", error=str(e))
if __name__=="__main__":
    response="""
```yaml
solutions:
- solution:
    name: "Probability Comparison"
    problem_variables: |
        - \( T \): Number of test cases
        - \( N \): Number of lines in the original solution
        - \( P \): Probability of typing a line correctly
        - \( Q \): Probability of typing a line correctly in the reduced solution
        - \( P_{new} \): New probability of typing a line correctly to match the reduced solution's chance of success
    
    approach_summary: |
        1. Core algorithm selection rationale: We need to compare the probability of success for two scenarios - typing \(N\) lines with probability \(P\) and typing \(N-1\) lines with probability \(Q\). The goal is to find \(Q\) such that the probability of success in both scenarios is equal.
        2. Key data structure choices: No additional data structures are needed beyond input variables.
        3. Mathematical foundation: The probability of success for typing \(N\) lines is \(P^N\), and for \(N-1\) lines is \(Q^{N-1}\). We need to solve for \(Q\) such that \(P^N = Q^{N-1}\).
        4. Optimization approach: We can solve the equation \(Q = P^{N/(N-1)}\) directly.
    
    detailed_solution: |
        1. Initialization
        - Input processing: Read \(T\), then for each test case, read \(N\) and \(P\).
        - Variable initialization: Initialize \(Q\) and \(P_{new}\).
        
        2. Core Algorithm
        - For each test case:
            - Calculate \(Q = P^{N/(N-1)}\)
            - Calculate \(P_{new} = Q^{N/(N-1)}\)
        
        3. Output Generation
        - Print the difference \(P_{new} - P\) for each test case.
    
    mathematical_foundation: |
        1. Formal Definitions:
        - \(P\): Probability of typing a line correctly
        - \(Q\): Probability of typing a line correctly in the reduced solution
        - \(P_{new}\): New probability of typing a line correctly to match the reduced solution's chance of success
        
        2. Key Formulas:
        - \(P^N = Q^{N-1}\)
        - Solving for \(Q\): \(Q = P^{N/(N-1)}\)
        - Solving for \(P_{new}\): \(P_{new} = Q^{N/(N-1)} = (P^{N/(N-1)})^{N/(N-1)} = P^N\)
        
        3. Numerical Analysis:
        - Stability proof: The formula \(Q = P^{N/(N-1)}\) is well-defined for \(0 < P < 1\) and \(N > 1\).
        - Error bounds: The error in \(Q\) and \(P_{new}\) is negligible due to the properties of exponents and floating-point arithmetic.
        - Precision requirements: Standard floating-point precision is sufficient for this problem.

    correctness_proof: |
        1. Invariants:
        - \(P\) and \(N\) are always positive integers within the given constraints.
        
        2. Termination:
        - The loop runs \(T\) times, where \(T\) is bounded by 100.
        
        3. Edge Cases:
        - \(P = 1\): \(P_{new} = 1\) for any \(N\).
        - \(N = 2\): \(Q = P^{2/1} = P^2\) and \(P_{new} = (P^2)^{2/1} = P^4\).
    
    complexity_analysis: |
        1. Time Complexity:
        - Overall: O(T)
        - Breakdown by component:
            - Input reading: O(T)
            - Calculation of \(Q\) and \(P_{new}\): O(1)
            - Output writing: O(T)
        - Worst-case scenarios: \(T = 100\)
        - Amortized analysis: Constant time per test case.
        
        2. Space Complexity:
        - Overall: O(1)
        - Auxiliary space: O(1)
        - Stack space: O(1)
        - Cache considerations: None
        
        3. Numerical Complexity:
        - Precision requirements: Standard floating-point precision is sufficient.
        - Error propagation: Errors in floating-point arithmetic are negligible.
        - Stability analysis: The formula \(Q = P^{N/(N-1)}\) is numerically stable for \(0 < P < 1\) and \(N > 1\).

    implementation_notes: |
        1. Code Structure: The code should be modular, with clear separation of input/output and core algorithm.
        2. Critical Optimizations: Use efficient floating-point arithmetic and avoid unnecessary calculations.
        3. Error Handling: Handle edge cases like \(P = 1\) and ensure input validity.
        4. Testing Strategy: Test with a variety of inputs, including edge cases and large values of \(N\).

    labels:
        - [algorithm_category]: Mathematical/formula-based solutions
        - [data_structures]: None
        - [mathematical_concepts]: Exponents, probability theory
        - [optimization_techniques]: None
```
"""
    parser = SolutionParser()

    solutions = parser.parse_solutions(response)
    print(solutions)