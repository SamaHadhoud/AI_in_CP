from pathlib import Path
input = Path('./sample_in.txt').read_text()

def check_winner(grid, player):
    # Check rows
    for row in grid:
        for i in range(4):
            if all(row[i + j] == player for j in range(4)):
                return True
    # Check columns
    for col in range(7):
        for i in range(3):
            if all(grid[i + j][col] == player for j in range(4)):
                return True
    # Check diagonals (top-left to bottom-right)
    for i in range(3):
        for j in range(4):
            if all(grid[i + k][j + k] == player for k in range(4)):
                return True
    # Check diagonals (top-right to bottom-left)
    for i in range(3):
        for j in range(3, 7):
            if all(grid[i + k][j - k] == player for k in range(4)):
                return True
    return False

def solve(input_data: str) -> str:
    lines = input_data.split('\n')
    T = int(lines[0])
    results = []

    for i in range(T):
        # Skip the empty line between test cases
        start_index = 2 + i * 7
        end_index = start_index + 7
        grid = lines[start_index:end_index]

        if check_winner(grid, 'C'):
            result = 'C'
        elif check_winner(grid, 'F'):
            result = 'F'
        elif any('C' in row for row in grid) and any('F' in row for row in grid):
            result = '?'
        else:
            result = '0'

        results.append(f"Case #{i + 1}: {result}")

    return '\n'.join(results)

output = solve(input)
Path('./full_out.txt').write_text(output)
