from pathlib import Path
input = Path('./full_in.txt').read_text()

from typing import List
from collections import deque
from heapq import heappop, heappush
from tqdm import tqdm

def solve(input_data: str) -> str:
    # Parse input
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    results = []

    for case in tqdm(range(1, T + 1)):
        try:
            N, K = map(int, lines[2 * case - 1].split())
            S = list(map(int, lines[2 * case].split()))

            # Sort travelers by their crossing time
            S.sort()

            # Initialize priority queue for the flashlight returns
            flashlight_queue = []

            # Initialize current time
            current_time = 0

            # Function to cross two travelers
            def cross_two(i, j):
                nonlocal current_time
                current_time += max(S[i], S[j])
                heappush(flashlight_queue, S[i] + S[j])

            # Function to cross one traveler
            def cross_one(i):
                nonlocal current_time
                current_time += S[i]
                heappush(flashlight_queue, S[i])

            # Initialize queue with the first two travelers
            queue = deque([0, 1])

            # Process the queue until all travelers are on the other side
            while queue or flashlight_queue:
                if queue:
                    # Cross the two fastest travelers
                    i, j = queue.popleft(), queue.popleft()
                    cross_two(i, j)
                else:
                    # Return the flashlight if there are still people left
                    if queue:
                        i = queue.popleft()
                        cross_one(i)

                # Add the next fastest traveler to the queue if possible
                if len(queue) < 2 and len(flashlight_queue) > 0:
                    next_flashlight_time = flashlight_queue[0]
                    next_traveler = next_flashlight_time - S[next_flashlight_time]
                    if next_traveler < N:
                        heappop(flashlight_queue)
                        queue.append(next_traveler)

                # Check if the current time exceeds K
                if current_time > K:
                    break

            # Check if all travelers can cross within K seconds
            if current_time <= K:
                results.append(f"Case #{case}: YES")
            else:
                results.append(f"Case #{case}: NO")

        except (ValueError, IndexError):
            # Handle incorrect input format or missing values
            results.append(f"Case #{case}: NO")

    return '\n'.join(results)

output = solve(input)
Path('./full_out.txt').write_text(output)
