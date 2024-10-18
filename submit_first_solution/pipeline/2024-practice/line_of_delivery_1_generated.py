from pathlib import Path
input = Path('./line_of_delivery_1.in').read_text()

def solve(input_data: str) -> str:
    result = []
    lines = input_data.strip().splitlines()
    T = int(lines[0])
    index = 1
    
    for _ in range(T):
        N, G = map(int, lines[index].split())
        energies = list(map(int, lines[index + 1:index + 1 + N]))
        index += 1 + N
        
        # Initialize stones with their initial positions and energies
        stones = [(i + 1, 0, e) for i, e in enumerate(energies)]
        
        # Simulate stone movement
        while True:
            moved = False
            for i in range(N):
                if stones[i][2] == 0:
                    continue
                
                # Move the stone one unit to the right
                stones[i] = (stones[i][0], stones[i][1] + 1, stones[i][2] - 1)
                moved = True
                
                # Check for collision with other stones
                for j in range(N):
                    if i != j and stones[j][1] == stones[i][1]:
                        stones[j] = (stones[j][0], stones[j][1], stones[j][2] + stones[i][2])
                        stones[i] = (stones[i][0], stones[i][1], 0)
                        moved = True
                        break
            
            if not moved:
                break
        
        # Determine the stone closest to the goal
        closest_distance = float('inf')
        closest_stone = -1
        
        for stone in stones:
            distance = abs(stone[1] - G)
            if distance < closest_distance:
                closest_distance = distance
                closest_stone = stone[0]
        
        result.append(f"Case #{_ + 1}: {closest_stone} {closest_distance}")
    
    return '\n'.join(result)

output = solve(input)
Path('./line_of_delivery_1_generated.out').write_text(output)
