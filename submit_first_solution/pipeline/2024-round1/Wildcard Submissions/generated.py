from pathlib import Path
input = Path('./full_in.txt').read_text()

from tqdm import tqdm

def solve(input_data: str) -> str:
    mod = 998244353
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    result = []

    for i in range(1, T+1):
        N = int(lines[i])
        trie = [[0, 1] for _ in range(101)]
        count = 0

        for j in range(i+1, i+N+1):
            s = lines[j]
            l = len(s)
            for k in range(l):
                if s[k] == '?':
                    for c in range(26):
                        count = (count + trie[k][0] * (trie[l-k-1][1] + 1)) % mod
                else:
                    count = (count + trie[k][0] * trie[l-k-1][1]) % mod

            for k in range(l+1):
                trie[k][0] = (trie[k][0] + trie[k][1]) % mod
                trie[k][1] = (trie[k][1] * (s[k] == '?' or s[k] == 'A' or s[k] == 'B' or s[k] == 'C')) % mod

        result.append(f"Case #{i}: {count}")

    return '\n'.join(result)

output = solve(input)
Path('./full_out.txt').write_text(output)
