from pathlib import Path
input = Path('./walk_the_line.in').read_text()

from collections import deque

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    results = []

    for t in range(1, T + 1):
        N, K = map(int, lines[t].split())
        S = list(map(int, lines[t + 1:t + N + 1]))

        S.sort()
        bridge = deque([S[-1], S[-2]])
        time = S[-1]

        flashlight = 0

        while len(bridge) > 1:
            if time + bridge[0] + bridge[1] <= K:
                time += bridge[0]
                bridge.popleft()
                bridge.popleft()
            else:
                time += bridge[0]
                bridge.popleft()
                bridge.appendleft(S[flashlight])
                flashlight += 1

        time += bridge[0]

        if time <= K:
            results.append(f"Case #{t}: YES")
        else:
            results.append(f"Case #{t}: NO")

    return '\n'.join(results)

output = solve(input)
Path('./walk_the_line_generated.out').write_text(output)
