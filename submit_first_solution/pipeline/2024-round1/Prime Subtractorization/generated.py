from pathlib import Path
input = Path('./full_in.txt').read_text()

from tqdm import tqdm
import math

def generate_primes(limit):
    sieve = [True] * (limit + 1)
    primes = []
    for num in tqdm(range(2, limit + 1), desc="Generating Primes"):
        if sieve[num]:
            primes.append(num)
            for multiple in range(num * num, limit + 1, num):
                sieve[multiple] = False
    return primes

def count_subtractorizations(primes, N):
    count = 0
    for i in range(len(primes)):
        if primes[i] > N:
            break
        for j in range(i, len(primes)):
            if primes[j] - primes[i] > N:
                break
            if primes[j] - primes[i] == N:
                count += 1
    return count

def solve(input_data: str) -> str:
    lines = input_data.strip().split("\n")
    T = int(lines[0])
    output = []
    
    for i in range(1, T + 1):
        N = int(lines[i])
        primes = generate_primes(2 * N)
        count = count_subtractorizations(primes, N)
        output.append(f"Case #{i}: {count}")
    
    return "\n".join(output)

output = solve(input)
Path('./full_out.txt').write_text(output)
