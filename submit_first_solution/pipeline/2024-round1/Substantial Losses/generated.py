from pathlib import Path
input = Path('./full_in.txt').read_text()

from typing import List
from math import gcd

def mod_inverse(a: int, m: int) -> int:
    return pow(a, m - 2, m)

def solve(input_data: str) -> str:
    MOD = 998244353
    test_cases = input_data.strip().split('\n')[1:]
    output = []
    
    for i, test_case in enumerate(test_cases, start=1):
        W, G, L = map(int, test_case.split())
        
        # Calculate the greatest common divisor of the difference
        diff = W - G
        lcm = L + 1
        g = gcd(diff, lcm)
        
        # Calculate the number of days
        days = diff // g
        
        # Calculate the result using modular inverse
        result = days * mod_inverse(2, MOD) % MOD
        output.append(f"Case #{i}: {result}")
    
    return '\n'.join(output)

output = solve(input)
Path('./full_out.txt').write_text(output)
