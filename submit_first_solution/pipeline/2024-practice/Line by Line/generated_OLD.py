from pathlib import Path
input = Path('./full_in.txt').read_text()

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    results = []

    for i in range(1, len(lines)):
        N, P = map(int, lines[i].split())
        P /= 100  # Convert P to a decimal

        # Calculate the probability of success for N lines with original P
        prob_N_lines = P ** N

        # Calculate the required increase in P
        required_increase = (prob_N_lines ** (1 / (N - 1))) * 100 - P * 100

        results.append(f"Case #{i}: {required_increase:.10f}")

    return '\n'.join(results)

output = solve(input)
Path('./full_out.txt').write_text(output)
