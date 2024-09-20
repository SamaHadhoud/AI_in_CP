To start, let’s look at the Sprague-Grundy function for a deck of arbitrary size \(k\):

\[\displaystyle g(k) = \text{mex}(\{ g(k - i) \oplus g(C_k) \mid i \in [ k - B_i, k - A_i ] \}) \]

In other words, to compute the value of \(g(k)\), we can perform the following steps:

1. Construct a set \(S\) of all values \(g(i)\), where \(i = k - B_i, ..., k - A_i\).
2. XOR each value of \(S\) with \(C_k\).
3. Find a mex (minimum excluded value) of \(S\) and assign it to \(g(k)\).

Since we need to compute \(g(k)\) for each \(k = 1..N\), we need a data structure that supports all the listed operations with \(\mathcal{O}(\log N)\) complexity. Such data structure exists – it’s a simple trie. Let’s represent each value of \(g(k)\) in its binary form and store binary string representations in the trie. Each node will have no more than two edges (one edge corresponds to \(0\), the other to \(1\)). For each node of the trie, let’s store the last value of \(k\) for which it has been updated.

Suppose for a certain value of \(k\) we want to find the mex of all values \(g(x)\) from \(g(l)\) to \(g(k)\). We can descend from the root of the trie to some leaf, each time going to the child which hasn’t been updated after \(l - 1\). If both child nodes satisfy this condition, we pass through \(0\)-edge. At any moment, if there is no corresponding edge, that means the node has never been updated.

Now we can modify this approach for finding the mex of all values \(g(x) \oplus y\) from \(g(l)\) to \(g(k)\) (for any value of \(y\)). To do so when both child nodes haven’t been updated after \(l - 1\) we will pass through \(0\)-edge only if the corresponding bit in \(y\) is \(0\), otherwise we will pass through \(1\)-edge.

The only modification we need now is to make this approach correct for an arbitrary right border \(r \ge l\). It can be done by simply making the trie persistent. This way we can always perform the previously explained approach on the version of the trie we had right after adding the value of \(g(r)\). The only extra price we pay is \(\log N\) times more memory usage.

Since any descent on a trie takes \(\mathcal{O}(\log N)\) steps, the overall complexity of the described approach is \(\mathcal{O}(N \log N)\).

When all values are computed, we can use them to find the value of \(K_2\) for each \(K_1\). First, \(K_2\) always exists and \(K_2 \le K_1\). If \(K_2 = K_1\), Bob can always win by mirroring Alice’s turns. Since we need Bob to win, we are interested in states where \(g(K_1) \oplus g(K_2)\) is zero and \(K_2\) is as small as possible. So all we need to do is find the minimum \(K_2\) such that \(g(K_1) = g(K_2)\). This can be done in \(\mathcal{O}(N)\) time and doesn’t affect the final \(\mathcal{O}(N \log N)\) complexity of the solution.

