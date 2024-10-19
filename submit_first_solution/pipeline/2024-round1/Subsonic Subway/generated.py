from pathlib import Path
input = Path('./full_in.txt').read_text()

from tqdm import tqdm
import math

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    output = []

    for i in tqdm(range(1, T+1), desc="Processing cases"):
        N = int(lines[i])
        intervals = []
        for j in range(i+1, i+N+1):
            A, B = map(int, lines[j].split())
            intervals.append((A, B))

        # Sort intervals by their end times
        intervals.sort(key=lambda x: x[1])

        min_speed = -1
        prev_end = 0
        for start, end in intervals:
            # Check if the previous end time is greater than the current start time
            if prev_end > start:
                min_speed = -1
                break
            else:
                # Calculate the minimum speed required to reach the current station
                min_speed = max(min_speed, end / (start + 1))
                prev_end = end

        output.append(f"Case #{i}: {min_speed:.6f}")

    return '\n'.join(output)

output = solve(input)
Path('./full_out.txt').write_text(output)
