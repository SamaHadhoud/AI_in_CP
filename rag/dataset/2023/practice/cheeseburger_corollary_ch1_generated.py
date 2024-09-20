from pathlib import Path
input = Path('./cheeseburger_corollary_ch1.in').read_text()

from tqdm import tqdm

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    output = []

    for i in tqdm(range(1, T+1), desc="Processing"):
        S, D, K = map(int, lines[i].split())
        total_buns = 2*S + 2*D
        total_patties = S + 2*D
        total_cheeses = S + 2*D

        if total_buns >= K and total_patties >= K and total_cheeses >= K:
            output.append(f"Case #{i}: YES")
        else:
            output.append(f"Case #{i}: NO")

    return '\n'.join(output)

output = solve(input)
Path('./cheeseburger_corollary_ch1_generated.out').write_text(output)
