from pathlib import Path
from tqdm import tqdm

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    tt = int(lines[0])
    current_line = 1
    result = []
    MD = 998244353
    
    for case in tqdm(range(1, tt + 1)):
        w, g, l = map(int, lines[current_line].split())
        
        a = (2 * l + 1) % MD
        b = (w - g) % MD
        ans = (a * b) % MD
        
        result.append(f"Case #{case}: {ans}")
        current_line += 1
    
    return '\n'.join(result)

input = Path('./full_in.txt').read_text()
output = solve(input)
Path('./full_out.txt').write_text(output)