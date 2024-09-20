from pathlib import Path
input = Path('./dim_sum_delivery.in').read_text()

from tqdm import tqdm

def solve(input_data: str) -> str:
    output = []
    test_cases = input_data.strip().split('\n')[1:]
    for i, test_case in enumerate(tqdm(test_cases, desc="Processing"), start=1):
        R, C, A, B = map(int, test_case.split())
        if (R - 1) % A == 0 and (C - 1) % B == 0:
            output.append(f"Case #{i}: YES")
        else:
            output.append(f"Case #{i}: NO")
    return '\n'.join(output)

output = solve(input)
Path('./dim_sum_delivery_generated.out').write_text(output)
