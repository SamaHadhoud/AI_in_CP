from pathlib import Path
input = Path('./line_of_delivery_2.in').read_text()

def simulate_stones(N, G, stones):
    positions = [0] * (N + 2)
    
    for i in range(N):
        x = 0
        energy = stones[i]
        
        while energy > 0:
            if x + 1 < N + 2 and positions[x + 1] == 0:
                positions[x + 1] = energy
                energy = 0
            elif x + 1 < N + 2 and positions[x + 1] > 0:
                energy += positions[x + 1]
                positions[x + 1] = 0
                x += 1
                energy -= 1
            elif x + 1 == N + 2:
                positions[x + 1] = energy
                energy = 0
            x += 1
            energy -= 1
    
    return positions

def find_closest_stone(N, G, positions):
    closest = (None, float('inf'))
    
    for i in range(N + 1):
        if positions[i] == 0:
            continue
        distance = abs(i + positions[i] - G)
        if distance < closest[1] or (distance == closest[1] and i < closest[0]):
            closest = (i + 1, distance)
    
    return closest

def solve(input_data: str) -> str:
    results = []
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    index = 1
    
    for t in range(T):
        N, G = map(int, lines[index].split())
        index += 1
        stones = [int(lines[index + i]) for i in range(N)]
        index += N
        
        positions = simulate_stones(N, G, stones)
        closest = find_closest_stone(N, G, positions)
        
        results.append(f"Case #{t + 1}: {closest[0]} {closest[1]}")
    
    return '\n'.join(results)

output = solve(input)
Path('./line_of_delivery_2_generated.out').write_text(output)
