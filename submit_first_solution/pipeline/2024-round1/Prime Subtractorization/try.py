from pathlib import Path
from tqdm import tqdm

def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# def count_n_subtractorizations(N):
#     """Count the number of N-subtractorizations."""
#     count = 0
#     for p in range(2, N + 1):
#         if is_prime(p):
#             for a in range(2, N + 1):
#                 if is_prime(a) and a > p:
#                     b = a - p
#                     if b > 0 and b <= N and is_prime(b):
#                         count += 1
#                         break  # Each prime p can only be part of one pair
#     return count

def count_n_subtractorizations(N):
    primes = [p for p in range(2, N + 1) if is_prime(p)]
    prime_differences = set()  # Use set for uniqueness
    
    for i in range(len(primes)):
        for j in range(i + 1, len(primes)):
            diff = abs(primes[j] - primes[i])
            if is_prime(diff):
                prime_differences.add(diff)
    
    return len(prime_differences)

def solve(input_data: str) -> str:
    """Solve the problem using the provided input data."""
    input_lines = input_data.strip().split('\n')
    T = int(input_lines[0])
    results = []

    for t in range(1, T + 1):
        N = int(input_lines[t])
        count = count_n_subtractorizations(N)
        results.append(f"Case #{t}: {count}")

    return '\n'.join(results)

input = Path('./full_in.txt').read_text()
output = solve(input)
Path('./full_out_bestrefine.txt').write_text(output)