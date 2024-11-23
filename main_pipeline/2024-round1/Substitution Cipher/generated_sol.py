from pathlib import Path
from tqdm import tqdm

class Mint:
    MOD = 998244353
    
    def __init__(self, v):
        self.v = v % self.MOD if v >= 0 else ((-v) % self.MOD * (self.MOD - 1)) % self.MOD
    
    def __add__(self, other):
        if isinstance(other, int):
            other = Mint(other)
        return Mint(self.v + other.v)
    
    def __str__(self):
        return str(self.v)
    
    def __int__(self):
        return self.v

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    current_line = 0
    tt = int(lines[current_line])
    result = []
    
    def get_next_line():
        nonlocal current_line
        current_line += 1
        return lines[current_line]
    
    for case in tqdm(range(1, tt + 1)):
        s, k = get_next_line().split()
        k = int(k)
        n = len(s)
        
        # Create t by replacing ? with 1
        t = ''.join('1' if c == '?' else c for c in s)
        
        # Calculate dp
        dp = [Mint(0)] * (n + 1)
        dp[0] = Mint(1)
        can2 = [False] * n
        
        for i in range(1, n + 1):
            if t[i-1] != '0':
                dp[i] += dp[i-1]
            if i >= 2:
                if t[i-2] == '1':
                    dp[i] += dp[i-2]
                    can2[i-2] = True
                if t[i-2] == '2' and t[i-1] <= '6':
                    dp[i] += dp[i-2]
                    can2[i-2] = True
        
        # Calculate f and fail arrays
        f = [[0] * 10 for _ in range(n)]
        fail = [[[False] * 10 for _ in range(10)] for _ in range(n)]
        
        for i in range(n-1, -1, -1):
            low = high = int(s[i]) if s[i] != '?' else 1
            if s[i] == '?':
                high = 9
                
            for d in range(low, high + 1):
                if i == n - 1:
                    f[i][d] = 1
                    continue
                    
                for p in range(10):
                    if can2[i] and (i + 2 == n or s[i + 2] != '0'):
                        if not (d == 1 or (d == 2 and p <= 6)):
                            fail[i][d][p] = True
                            continue
                    f[i][d] = min(k, f[i][d] + f[i + 1][p])
        
        # Construct the result string
        result_s = list(s)
        k_remaining = k
        
        for i in range(n):
            low = high = int(s[i]) if s[i] != '?' else 1
            if s[i] == '?':
                high = 9
                
            found = False
            for d in range(high, low - 1, -1):
                if i > 0:
                    q = int(result_s[i-1])
                    if fail[i-1][q][d]:
                        continue
                        
                if k_remaining > f[i][d]:
                    k_remaining -= f[i][d]
                else:
                    result_s[i] = str(d)
                    found = True
                    break
            
            assert found
        
        result_str = ''.join(result_s)
        result.append(f"Case #{case}: {result_str} {dp[n]}")
    
    return '\n'.join(result)

input = Path('./full_in.txt').read_text()
output = solve(input)
Path('./full_out.txt').write_text(output)