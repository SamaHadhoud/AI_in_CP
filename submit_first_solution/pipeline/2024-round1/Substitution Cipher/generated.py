from pathlib import Path
input = Path('./full_in.txt').read_text()

from tqdm import tqdm
import sys

MOD = 998244353

def solve(input_data: str) -> str:
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    test_cases = lines[1:]

    results = []
    for i, test_case in enumerate(tqdm(test_cases, desc='Processing test cases')):
        E, K = test_case.split()
        K = int(K)
        N = len(E)
        E = E.replace('?', '0')

        dp = [[[0, 0] for _ in range(15)] for _ in range(N+1)]
        dp[0][0][1] = 1

        for i in range(N):
            for j in range(14):
                for k in range(2):
                    if dp[i][j][k] == 0: continue
                    for digit in range(10):
                        if digit == 0 and i == 0: continue
                        if digit == 0 and i > 0 and E[i-1] == '0': continue
                        if digit > 0 and j + 1 > 10: continue
                        if k == 1 and digit > int(E[i]): continue
                        dp[i+1][j + (digit > 0)][1] = (dp[i+1][j + (digit > 0)][1] + dp[i][j][k]) % MOD
                        if k == 1 or digit < int(E[i]):
                            dp[i+1][j + 1][0] = (dp[i+1][j + 1][0] + dp[i][j][k]) % MOD

        total = sum(dp[N][j][k] for j in range(11) for k in range(2)) % MOD
        count = [0] * (total + 1)
        for j in range(11):
            for k in range(2):
                count[dp[N][j][k]] += 1

        for i in range(total, 0, -1):
            count[i] = (count[i] + count[i + 1]) % MOD

        ans = ''
        for i in range(N, 0, -1):
            for j in range(10):
                if j == 0 and E[i-1] == '0': continue
                if dp[i-1][j][0] < K and K <= dp[i-1][j][0] + count[dp[i-1][j][1] - dp[i-1][j][0]]:
                    ans = str(j) + ans
                    K -= dp[i-1][j][0]
                    if j > 0: K -= 1
                    break
                K %= count[dp[i-1][j][1] - dp[i-1][j][0]]

        results.append("Case #{}: {} {}".format(i+1, ans, total))

    return '\n'.join(results)

output = solve(input)
Path('./full_out.txt').write_text(output)
