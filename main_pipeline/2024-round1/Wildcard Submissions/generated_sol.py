from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

def solve(input_data: str) -> str:
    P = 998244353
    lines = input_data.strip().split('\n')
    current_line = 0
    
    def read_line() -> str:
        nonlocal current_line
        line = lines[current_line]
        current_line += 1
        return line
    
    T = int(read_line())
    output = []
    
    for ti in range(1, T + 1):
        n = int(read_line())
        s = [read_line() for _ in range(n)]
        
        L = max(len(si) for si in s)
        
        # Initialize dp with full mask and count 1
        dp: List[Tuple[int, int]] = [(((1 << n) - 1), 1)]
        
        ans = 1
        for i in range(L):
            ndp = []
            qm = 0  # question mark mask
            cm = [0] * 26  # character masks
            
            # Build masks
            for j in range(n):
                if i >= len(s[j]):
                    continue
                if s[j][i] == '?':
                    qm |= 1 << j
                else:
                    cm[ord(s[j][i]) - ord('A')] |= 1 << j
            
            # Get character list and count remaining characters
            clst = []
            rest = 0
            for c in range(26):
                if cm[c]:
                    clst.append(cm[c])
                else:
                    rest += 1
            
            def add_to(mask: int, v: int) -> None:
                nonlocal ans
                if not mask or not v:
                    return
                ndp.append((mask, v))
                ans = (ans + v) % P
            
            # Process each state in dp
            for mask, v in dp:
                add_to(mask & qm, v * rest)
                for cm_val in clst:
                    add_to(mask & (qm | cm_val), v)
            
            # Sort and merge similar masks
            ndp.sort()
            dp = []
            for mask, v in ndp:
                if not dp or dp[-1][0] != mask:
                    dp.append((mask, 0))
                dp[-1] = (dp[-1][0], (dp[-1][1] + v) % P)
        
        output.append(f"Case #{ti}: {ans}")
    
    return '\n'.join(output)

input = Path('./full_in.txt').read_text()
output = solve(input)
Path('./full_out.txt').write_text(output)