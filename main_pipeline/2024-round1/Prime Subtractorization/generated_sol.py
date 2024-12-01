from pathlib import Path
input = Path('./prime_subtractorization_input.txt').read_text()

def sieve_of_eratosthenes(n):
    is_prime = [True] * (n + 1)
    p = 2
    while p * p <= n:
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1
    return [p for p in range(2, n + 1) if is_prime[p]]

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    current_line = 0
    T = int(lines[current_line])
    current_line += 1
    result = []
    
    for t in range(1, T + 1):
        N = int(lines[current_line])
        current_line += 1
        
        # Generate all primes less than or equal to N
        primes = sieve_of_eratosthenes(N)
        prime_set = set(primes)
        
        # Count N-subtractorizations
        count = 0
        for p in primes:
            if p + 2 in prime_set:
                count += 1
        
        # Add 1 for the pair (2, 3) if it's within the range
        if 5 <= N:
            count += 1
        
        result.append(f"Case #{t}: {count}")
    
    return '\n'.join(result)

# Example usage:
input_data = """
2
5
8
"""
print(solve(input_data))

output = solve(input)
Path('./full_out.txt').write_text(output)
