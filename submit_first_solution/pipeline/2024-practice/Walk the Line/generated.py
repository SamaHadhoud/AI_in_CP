from pathlib import Path
input = Path('./full_in.txt').read_text()

from io import StringIO

def solve(input_data: str) -> str:
    # Read input data
    input = StringIO(input_data)
    
    results = []
    
    # Number of test cases
    T = int(input.readline().strip())
    
    for case in range(1, T + 1):
        # Read N and K
        N, K = map(int, input.readline().strip().split())
        
        # Read the times for each traveler
        times = [int(input.readline().strip()) for _ in range(N)]
        
        # Sort the times
        times.sort()
        
        # Initialize variables
        total_time = 0
        left, right = 0, N - 1
        
        while left <= right:
            # Fastest and slowest travelers
            fastest = times[left]
            slowest = times[right]
            
            # If there's only one traveler left, he crosses alone
            if left == right:
                total_time += fastest
                break
            
            # Fastest crosses with slowest
            total_time += slowest
            # Slowest returns with flashlight
            total_time += fastest
            # Fastest crosses alone
            total_time += fastest
            
            # Move pointers
            left += 1
            right -= 1
        
        # Check if the total time is within K
        if total_time <= K:
            results.append(f"Case #{case}: YES")
        else:
            results.append(f"Case #{case}: NO")
    
    return "\n".join(results)

# Example usage:
# input_data = """6
# 3 1000000000
# 1000000000
# 1000000000
# 1000000000
# 4 17
# 1
# 2
# 5
# 10
# 4 4
# 1
# 2
# 5
# 10
# 2 22
# 22
# 22
# 1 100
# 12
# 1 10
# 12"""
# print(solve(input_data))

output = solve(input)
Path('./full_out.txt').write_text(output)
