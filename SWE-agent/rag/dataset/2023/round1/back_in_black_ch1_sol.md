First, note that pressing any button twice is equivalent to doing nothing. We can read in all \(Q\) button presses into a boolean array \(P_{1..N}\), where \(P_i\) stores for button \(i\) the total number of presses modulo \(2\).

Next, we'll need to apply this array of presses to our original array \(S_{1..N}\). Naively, it would look something like:
```
for (int i = 1; i <= N; i++) {
    if (P[i]) {
        for (int j = i; j <= N; j += i) {
            S[j] ^= 1;
        }
    }
}
```

For \(i = 1\), the inner loop runs for \(N\) steps. For \(i = 2\), the inner loop runs for \(N / 2\) steps. So forth, until a total of \(N + \frac{N}{2} + \frac{N}{3} + ... + \frac{N}{N}\) steps. Factoring out \(N\), we get \(N*(1 + \frac{1}{2} + ... + \frac{1}{N})\). The second factor is the [harmonic series](https://en.wikipedia.org/wiki/Harmonic_series_(mathematics)), which converges to roughly \(\log_2 (N)\). Thus, such a nested loop would only take \(\mathcal{O}(N \log N)\) time.

After we've applied the presses to \(S\), we now need to convert every \(1\) to \(0\) with the minimum number of button presses. Consider the following greedy algorithm:

```
int num_presses = 0;
for (int i = 1; i <= N; i++) {
    if (S[i]) {
      num_presses++;
      for (int j = i; j <= N; j += i) {
        S[j] ^= 1;
      }
    }
}
```

We scan from left to right, and each time we see a \(1\), we'll simply press the button and apply the changes to the rest of the array. In fact, this is not a greedy method, but the only method we can convert all the buttons to \(0\). A button can only be changed by pressing itself, or some button before it. If it were not optimal to clear the leftmost \(1\) by pressing it directly, then we would have to press another \(1\) before it, which contradicts the former \(1\) being leftmost.

The time complexity of this approach algorithm is analyzed similarly as the original application of the \(Q\) button presses, so the overall running time is \(\mathcal{O}(N \log N)\).
