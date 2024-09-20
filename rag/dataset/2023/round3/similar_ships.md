The Flying Dutchman is making a killing giving out haunted house tours of his ghost ship. The ship has \(N\) cabins, numbered from \(1\) to \(N\), connected by corridors in the shape of an unrooted tree \(t_1\). For each \(i=2..N\), there is a corridor that connects cabin \(i\) to cabin \(P_{i}\).

Mr. Krabs noticed this business opportunity, and wants to make his own ship, also with \(N\) cabins, connected as a yet-to-be-determined tree \(t_2\). To avoid any lawsuits from the Dutchman, he would like to make his ship as different as possible.

In particular, let \(S(t)\) be the multiset of all subtrees of tree \(t\), where a subtree is any non-empty subset of cabins that are connected. For two multisets of trees \(S_1\) and \(S_2\), let \(\text{similarity}(S_1, S_2)\) be the number of trees in \(S_2\) that are [isomorphic](https://en.wikipedia.org/wiki/Graph_isomorphism) to at least one tree in \(S_1\). Note that \(\text{similarity}(S_1, S_2)\) and \(\text{similarity}(S_2, S_1)\) are not necessarily equal.

Help Mr. Krabs find the minimum possible value of \(\text{similarity}(S(t_1), S(t_2))\) over all trees \(t_2\) of size \(N\). As this value may be large, output it modulo \(1{,}000{,}000{,}007\).


# Constraints

\(1 \leq T \leq 35\)
\(2 \leq N \leq 1{,}000{,}000\)
\(1 \leq P_i \leq N\)

The sum of \(N\) across all test cases is at most \(10{,}000{,}000\).


# Input Format

Input begins with an integer \(T\), the number of test cases. Each case has two lines. The first contains the integer \(N\), and the second contains the \(N-1\) integers \(P_2\) through \(P_N\).


# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by the minimum possible similarity if \(t_2\) is chosen optimally, modulo \(1{,}000{,}000{,}007\).


# Sample Explanation

The first sample case is depicted below. One possible tree that minimizes the similarity score is the same tree given in the input (i.e. choose \(t_2 = t_1\)). Then, \(S(t_2)\) consists of the \(6\) subtrees: \(\{1\}\), \(\{2\}\), \(\{3\}\), \(\{1, 2\}\), \(\{2, 3\}\), and \(\{1, 2, 3\}\). Each subtree is isomorphic to a subtree of \(t_1\), so the similarity score is \(6\).

{{PHOTO_ID:720027639598818|WIDTH:280}}

In the second case, one possible \(t_2\) that minimizes the similarity score is depicted below. This tree has \(24\) different subtrees, of which \(21\) are isomorphic to a subtree of the given \(t_1\). For example, the subtree \(\{2, 3, 4, 6\}\) of this \(t_2\) is isomorphic to subtree \(\{2, 3, 4, 5\}\) of the given \(t_1\), but the subtree \(\{1, 2, 3, 6\}\) of this \(t_2\) is not isomorphic to any subtree of \(t_1\).

{{PHOTO_ID:854841726123175|WIDTH:350}}
