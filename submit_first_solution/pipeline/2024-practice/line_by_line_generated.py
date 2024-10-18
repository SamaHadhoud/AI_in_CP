from pathlib import Path
input = Path('./line_by_line_input.txt').read_text()

def solve(input_data: str) -> str:
    output = []
    cases = input_data.strip().split('\n')[1:]
    
    for i, case in enumerate(cases, start=1):
        N, P = map(int, case.split())
        P /= 100.0  # Convert P to a decimal
        
        # Calculate the new P that matches the success rate of N-1 lines with original P
        new_P = P ** ((N-1)/N)
        
        # Calculate the increase in P needed
        increase = new_P - P
        
        # Convert the increase back to a percentage
        increase *= 100
        
        output.append(f"Case #{i}: {increase:.6f}")
    
    return '\n'.join(output)

output = solve(input)
Path('./line_by_line_generated.out').write_text(output)
