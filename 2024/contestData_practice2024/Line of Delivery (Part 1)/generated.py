from pathlib import Path
input = Path('./full_in.txt').read_text()

from tqdm import tqdm

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    results = []

    for sample in tqdm(range(1, T + 1)):
        line = lines[2 * sample - 1]
        N, G = map(int, line.split())
        energies = list(map(int, lines[2 * sample + 1].split()))

        stones = [(0, E, j) for j, E in enumerate(energies)]
        positions = [0] * N

        while stones:
            stones.sort(key=lambda x: x[0])  # Sort stones by their current position
            pos, energy, index = stones[0]
            if pos + energy >= G:
                stones[0] = (pos + energy, 0, index)
                break

            stones[0] = (pos + 1, energy - 1, index)

            for j in range(1, len(stones)):
                next_pos, next_energy, next_index = stones[j]
                if stones[0][0] == next_pos and next_energy > 0:
                    stones[0] = (next_pos, 0, index)
                    stones[j] = (next_pos, next_energy + energy, next_index)
                    break

        for j, (pos, energy, index) in enumerate(stones):
            positions[index] = pos

        min_distance = float('inf')
        closest_stone = -1
        for j, pos in enumerate(positions):
            distance = abs(pos - G)
            if distance < min_distance or (distance == min_distance and j < closest_stone):
                min_distance = distance
                closest_stone = j

        results.append(f"Case #{sample}: {closest_stone + 1} {min_distance}")

    return '\n'.join(results)

output = solve(input)
Path('./full_out.txt').write_text(output)
