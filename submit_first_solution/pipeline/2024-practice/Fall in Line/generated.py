from pathlib import Path
input = Path('./full_in.txt').read_text()

from tqdm import tqdm

def solve(input_data: str) -> str:
    # Split the input data into lines
    lines = input_data.strip().split('\n')
    
    # Initialize the result list
    result = []
    
    # Process each test case
    for i in tqdm(range(1, len(lines), 2)):
        # Parse the number of ants
        N = int(lines[i])
        
        # Parse the coordinates of the ants
        ants = [tuple(map(int, lines[i + j + 1].split())) for j in range(N)]
        
        # Function to calculate the number of ants to move to align them on a line
        def min_moves_to_line(ants):
            from collections import Counter
            from itertools import combinations
            
            # Helper function to calculate the slope between two points
            def slope(p1, p2):
                if p1[0] == p2[0]:
                    return float('inf')  # Vertical line
                return (p1[1] - p2[1]) / (p1[0] - p2[0])
            
            # Count the frequency of each slope
            slope_counts = Counter(slope(p1, p2) for p1, p2 in combinations(ants, 2))
            
            # The minimum number of ants to move is the total number of ants minus the maximum frequency of any slope
            return N - max(slope_counts.values())
        
        # Calculate the minimum number of moves for this test case
        moves = min_moves_to_line(ants)
        
        # Append the result for this test case
        result.append(f"Case #{(i + 1) // 2}: {moves}")
    
    # Join the results into a single string and return
    return '\n'.join(result)

output = solve(input)
Path('./full_out.txt').write_text(output)
