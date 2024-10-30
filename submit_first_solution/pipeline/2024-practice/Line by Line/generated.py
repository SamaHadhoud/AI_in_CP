from pathlib import Path
input = Path('./full_in.txt').read_text()

import math

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    results = []

    for i in range(1, T + 1):
        N, P = map(int, lines[i].split())
        P /= 100.0

        new_P = P ** (1 + 1 / (N - 1))
        increase_in_P = new_P - P

        increase_in_P *= 100.0

        results.append(f"Case #{i}: {increase_in_P:.10f}")

    return '\n'.join(results)

output = solve(input)
Path('./full_out.txt').write_text(output)
