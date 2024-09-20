In this part, it won't be efficient enough to directly simulate zeroing out the array in \(\mathcal{O}(N \log N)\) after each of the \(Q\) queries. Instead, we can track the number of presses needed as we go.

Before your brother presses anything, we can first compute the array \(P_{1..N}\) indicating whether a press is needed to zero out the initial array \(S_{1..N}\), using the same greedy algorithm from chapter 1. We can then keep a counter of the minimum number of presses needed. This part takes \(\mathcal{O}(N \log N)\) time.

From this point, if your brother presses some button \(i\) while previously \(P_i = 1\), then he has done us a favor, and we would no longer need to press that button to zero out the array. We can set \(P_i = 0\) and decrement our counter. Otherwise, if he pressed \(i\) while \(P_i = 0\) then we would now need to set \(P_i = 1\) and increment our counter.

The key is that the counter always holds the optimal number of presses to zero out the array, so we increment the overall answer by the counter after each of the \(Q\) queries. Each query takes \(\mathcal{O}(1)\) to process, so overall running time is \(\mathcal{O}(N \log N + Q)\).
