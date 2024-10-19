from pathlib import Path
input = Path('./full_in.txt').read_text()

from tqdm import tqdm

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    output = ''
    
    for i in tqdm(range(1, len(lines), 2)):
        N = int(lines[i])
        G = int(lines[i + 1])
        energies = list(map(int, lines[i + 2:i + 2 + N]))
        
        positions = [0] * N
        current_energies = energies[:]
        
        for j in range(N):
            p = positions[j]
            E = current_energies[j]
            while E > 0:
                if p + 1 < N and positions[p + 1] == 0:
                    positions[p + 1] = p + 1
                    current_energies[p + 1] += E
                    current_energies[j] = 0
                    break
                p += 1
                E -= 1
            positions[j] = p
        
        distances = [G - pos for pos in positions]
        min_distance = min(distances)
        closest_stone = distances.index(min_distance)
        
        output += f'Case #{(i // 2) + 1}: {closest_stone + 1} {min_distance}\n'
    
    return output.strip()

output = solve(input)
Path('./full_out.txt').write_text(output)
