from pathlib import Path
input = Path('./fall_in_line.in').read_text()

from collections import defaultdict

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def find_slope(p1, p2):
    if p1[0] == p2[0]:
        return float('inf')  # Vertical line
    return (p1[1] - p2[1]) / (p1[0] - p2[0])

def count_collinear_points(points):
    slope_count = defaultdict(int)
    vertical_count = 0
    horizontal_count = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i][0] == points[j][0]:
                vertical_count += 1
            elif points[i][1] == points[j][1]:
                horizontal_count += 1
            else:
                slope = find_slope(points[i], points[j])
                slope_count[slope] += 1
    return max(slope_count.values(), default=0) + vertical_count + horizontal_count

def min_ants_to_move(points):
    n = len(points)
    max_collinear = 0
    for i in range(n):
        remaining_points = points[:i] + points[i+1:]
        max_collinear = max(max_collinear, count_collinear_points(remaining_points))
    return n - max_collinear

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    result = []

    index = 1
    for _ in range(T):
        N = int(lines[index])
        points = []
        for _ in range(N):
            x, y = map(int, lines[index + 1].split())
            points.append((x, y))
            index += 1
        result.append(f"Case #{_ + 1}: {min_ants_to_move(points)}")

    return '\n'.join(result)

output = solve(input)
Path('./fall_in_line_generated.out').write_text(output)
