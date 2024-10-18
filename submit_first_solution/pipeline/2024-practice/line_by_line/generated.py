from pathlib import Path
input = Path('./full_in.txt').read_text()

from tqdm import tqdm
import math

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    result = []
    
    for i in tqdm(range(1, len(lines))):
        N, P = map(int, lines[i].split())
        P /= 100  # Convert P to a probability (0 to 1)
        
        # Probability of success with N lines
        prob_N_lines = P ** N
        
        # Probability of success with N-1 lines
        prob_N_minus_1_lines = math.sqrt(prob_N_lines)
        
        # Find the required increase in P to make the probabilities equal
        required_increase = math.sqrt(P) - P
        
        # Convert P back to percentage
        required_increase *= 100
        
        result.append(f"Case #{i}: {required_increase:.12f}")
    
    return '\n'.join(result)

output = solve(input)
Path('./full_out.txt').write_text(output)
