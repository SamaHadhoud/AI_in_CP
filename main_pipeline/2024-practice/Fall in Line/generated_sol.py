from pathlib import Path
from tqdm import tqdm
import random
from math import sqrt
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

def collinear(a: Point, b: Point, c: Point) -> bool:
    # Calculate cross product to determine if points are collinear
    cross_product = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)
    return abs(cross_product) < 1e-10

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    current_line = 1
    result = []
    
    for case in tqdm(range(1, T + 1)):
        N = int(lines[current_line])
        ants = []
        
        # Read ant coordinates
        for i in range(N):
            x, y = map(int, lines[current_line + 1 + i].split())
            ants.append(Point(x, y))
            
        # Find maximum collinear points through random sampling
        ITERS = 100
        most_collinear = 0
        
        for _ in range(ITERS):
            # Pick two random distinct points
            a, b = random.sample(range(N), 2)
            point_a, point_b = ants[a], ants[b]
            
            # Count collinear points
            line_count = 2  # Starting with the two points we picked
            for i in range(N):
                if i != a and i != b and collinear(point_a, point_b, ants[i]):
                    line_count += 1
            
            most_collinear = max(most_collinear, line_count)
        
        result.append(f"Case #{case}: {N - most_collinear}")
        current_line += N + 1
        
    return '\n'.join(result) + '\n'

input = Path('./full_in.txt').read_text()
output = solve(input)
Path('./full_out.txt').write_text(output)