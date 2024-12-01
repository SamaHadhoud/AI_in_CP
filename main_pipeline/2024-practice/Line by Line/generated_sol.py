from pathlib import Path
input = Path('./full_in.txt').read_text()
def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    results = []
    
    for i in range(1, len(lines)):
        N, P = map(int, lines[i].split())
        p_original = P / 100.0
        p_new = p_original ** (1 - 1/N)
        increase = (p_new - p_original) * 100.0
        results.append(f"Case #{i}: {increase:.10f}")
    
    return '\n'.join(results)

output = solve(input)
Path('./full_out.out').write_text(output)
