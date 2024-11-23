from pathlib import Path
from tqdm import tqdm

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    tt = int(lines[0])
    current_line = 1
    result = []
    
    for case in tqdm(range(1, tt + 1)):
        n = int(lines[current_line])
        print(n)
        a = [0] * (n + 1)
        b = [0] * (n + 1)
        # current_line+=1
        for i in range(1, n+1):
            print(current_line+i)
            print(lines[current_line+i])
            a[i], b[i] = map(int, lines[current_line+i].split(' '))
        
        mn = 1
        mx = 1
        
        for i in range(2, n + 1):
            if a[i] * mn > a[mn] * i:
                mn = i
            if b[i] * mx < b[mx] * i:
                mx = i
        
        if a[mn] * mx > b[mx] * mn:
            ans = -1
        else:
            ans = mx / b[mx]
        
        result.append(f"Case #{case}: {ans if ans == -1 else f'{ans:.17f}'}")
        current_line += n +1
    
    return '\n'.join(result)

input = Path('./full_in.txt').read_text()
output = solve(input)
Path('./full_out.txt').write_text(output)