from pathlib import Path
input = Path('./sample_in.txt').read_text()

def is_peak(number):
    digits = [int(d) for d in str(number)]
    n = len(digits)
    if n % 2 == 0 or 0 in digits:
        return False
    k = n // 2
    if digits[:k] != list(range(1, k + 2)) or digits[k+1:] != list(range(k, 0, -1)):
        return False
    return True

def count_peaks_in_range(A, B):
    count = 0
    for number in range(max(A, 1), B + 1):
        if is_peak(number) and number % M == 0:
            count += 1
    return count

def solve(input_data: str) -> str:
    result = []
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    for i in range(1, T + 1):
        A, B, M = map(int, lines[i].split())
        count = count_peaks_in_range(A, B)
        result.append(f"Case #{i}: {count}")
    return '\n'.join(result)

output = solve(input)
Path('./full_out.txt').write_text(output)
