from pathlib import Path
input = Path('./full_in.txt').read_text()

from typing import List

def solve(input_data: str) -> str:
    from io import StringIO
    
    # Parse input
    input_lines = input_data.strip().split('\n')
    T = int(input_lines[0])
    results = []
    
    index = 1
    for _ in range(T):
        R, C, K = map(int, input_lines[index].split())
        index += 1
        burrows = [[0] * C for _ in range(R)]
        
        for i in range(R):
            burrows[i] = list(map(int, input_lines[index].split()))
            index += 1
        
        # Find all valid hops and their scores
        scores = []
        for i in range(R):
            for j in range(C):
                for ii in range(R):
                    for jj in range(C):
                        if burrows[i][j] != burrows[ii][jj]:
                            score = max(abs(ii - i), abs(jj - j))
                            scores.append(score)
        
        # Sort scores and find the K-th smallest
        scores.sort()
        Kth_smallest = scores[K - 1]
        
        # Store the result
        results.append(f"Case #{_ + 1}: {Kth_smallest}")
    
    return '\n'.join(results)

# Example usage
input_data = """4
1 3 3
1 1 2
1 4 12
1 2 3 4
2 2 5
1 2
2 1
2 3 17
1 1 2
1 2 2"""
print(solve(input_data))

output = solve(input)
Path('./full_out.txt').write_text(output)
