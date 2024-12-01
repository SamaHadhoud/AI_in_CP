from pathlib import Path
from tqdm import tqdm

def solve(input_data: str) -> str:
    # Parse input
    lines = input_data.strip().split('\n')
    t = int(lines[0])
    current_line = 1
    result = []
    
    # Process each test case
    for case in range(1, t + 1):
        N, G = map(int, lines[current_line].split())
        E = list(map(int, lines[current_line + 1: current_line + 1 + N]))
        
        # Sort the array
        E.sort()
        index = 0
        
        # Find the element closest to G
        for i in range(1, N):
            if abs(E[i] - G) <= abs(E[index] - G):
                index = i
        
        # Format output
        result.append(f"Case #{case}: {N - index} {abs(E[index] - G)}")
        current_line += (N+1)
    
    return '\n'.join(result) + '\n'

input = Path('./full_in.txt').read_text()
output = solve(input)
Path('./full_out.txt').write_text(output)