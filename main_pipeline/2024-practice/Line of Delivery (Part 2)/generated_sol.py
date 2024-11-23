from pathlib import Path
from tqdm import tqdm

def solve(input_data: str) -> str:
    # Split input into lines and remove any trailing whitespace
    lines = [line.strip() for line in input_data.strip().split('\n')]
    cur_line = 0
    
    # Read number of test cases
    T = int(lines[cur_line])
    cur_line += 1
    
    output = []
    # Process each test case
    for t in tqdm(range(1, T + 1)):
        # Read N and G
        N, G = map(int, lines[cur_line].split())
        cur_line += 1
        
        # Read E array - one number per line
        E = []
        for _ in range(N):
            E.append(int(lines[cur_line]))
            cur_line += 1
        
        # Adjust E values and create s array
        s = [E[i] - (N - i - 1) for i in range(N)]
        s.sort()
        
        # Add index to each element in s
        for i in range(N):
            s[i] += i
            
        # Find minimum pair (abs difference with G, -s[i])
        min_pair = (2e9, -1)  # Initialize with large value
        for i in range(N):
            curr_pair = (abs(s[i] - G), -s[i])
            if curr_pair < min_pair:
                min_pair = curr_pair
                
        # Count elements less than -second element of min_pair
        c = sum(1 for x in s if x < -min_pair[1])
        
        # Format output
        output.append(f"Case #{t}: {N-c} {min_pair[0]}")
    
    return '\n'.join(output)

input = Path('./full_in.txt').read_text()
output = solve(input)
Path('./full_out.txt').write_text(output)