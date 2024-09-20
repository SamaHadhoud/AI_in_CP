from pathlib import Path
input = Path('./cheeseburger_corollary_ch2.in').read_text()

from tqdm import tqdm

def solve(input: str) -> str:
    output = []
    lines = input.strip().split('\n')
    T = int(lines[0])
    for i in tqdm(range(1, T + 1)):
        A, B, C = map(int, lines[i].split())
        if C < A and C < B:
            output.append(f"Case #{i}: 0")
        else:
            K = (C // (A + B)) * 2
            if C % (A + B) >= A:
                K += 1
            output.append(f"Case #{i}: {K}")
    return '\n'.join(output)

output = solve(input)
Path('./cheeseburger_corollary_ch2_generated.out').write_text(output)
