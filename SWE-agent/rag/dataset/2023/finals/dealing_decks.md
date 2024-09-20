Alice and Bob like to play cards on their lunch break. Their favorite card game starts with two decks on the table containing \(K_1\) and \(K_2\) cards. Players take turns, with Alice going first. Each turn is in two parts:

1. The player selects a deck of cards from the table. Let \(k\) be the number of cards in that deck. They then remove somewhere between \(A_k\) and \(B_k\) \((1 \le A_k \le B_k \le k)\), inclusive, cards from this deck.
2. The player then puts a new deck of exactly \(C_k\) \((0 \le C_k < k)\) cards on the table. Here, \(C_k = 0\) means no deck is added.

The player who takes the last card wins.

For each possible value of \(K_1\) from \(1\) to a given value \(N\), find the minimum possible value of \(K_2\) so that Bob wins the game if both players play optimally. If there is no such \(K_2\) between \(1\) and \(N\), then \(K_2 = -1\). Output the sum of \(K_2\) across every \(K_1 = 1..N\).

To reduce the input size, you will not be given \(A_{1..N}\), \(B_{1..N}\), and \(C_{1..N}\) directly. You must instead generate them using parameters \(X_a\), \(Y_a\), \(Z_a\), \(X_b\), \(Y_b\), \(Z_b\), \(X_c\), \(Y_c\), and \(Z_c\) and the algorithm below:

\(P_a[0] := 0\)
\(P_b[0] := 0\)
\(P_c[0] := 0\)
\(\text{for each } i := 1..N\):
\(\;\;\;\;\; P_a[i] := (P_a[i - 1] * X_a + Y_a) \text{ mod } Z_a\)
\(\;\;\;\;\; P_b[i] := (P_b[i - 1] * X_b + Y_b) \text{ mod } Z_b\)
\(\;\;\;\;\; P_c[i] := (P_c[i - 1] * X_c + Y_c) \text{ mod } Z_c\)
\(\;\;\;\;\; A[i] := \min(i, 1 + P_a[i])\)
\(\;\;\;\;\; B[i] := \max(A[i], i - P_b[i])\)
\(\;\;\;\;\; C[i] := \min(i - 1, P_c[i])\)

Note that for any \(i\), the algorithm guarantees \(1 \le A_i \le B_i \le i\) and \(0 \le C_i < i\).

# Constraints

\(1 \le T \le 45\)
\(1 \le N \le 2{,}000{,}000\)
\(1 \le X_a, Y_a, X_b, Y_b, X_c, Y_c \le 1{,}000{,}000{,}000\)
\(1 \le Z_a, Z_b, Z_c \le N\)

The sum of \(N\) across all test cases is at most \(50{,}000{,}000\).

# Input Format

Input begins with a single integer \(T\), the number of test cases. For each case, first there is a line containing a single integer \(N\). Then, there is a line containing integers \(X_a\), \(Y_a\), \(Z_a\), \(X_b\), \(Y_b\), \(Z_b\), \(X_c\), \(Y_c\), and \(Z_c\).

# Output Format

For the \(i\)th case, output `"Case #i: "` followed by a single integer, the sum of the minimum \(K_2\) so that Bob has a guaranteed winning strategy, for every \(K_1 = 1..N\).

# Sample Explanation

In the first sample case:

\( \begin{array}{c|c|c|c|c}\underline{K_1}&\underline{A[K_1]}&\underline{B[K_1]}&\underline{C[K_1]}&\underline{K_2} \\ 1&1&1&0&1 \\ 2&1&2&0&2 \\ 3&1&3&2&1 \\ 4&1&4&0&4 \end{array} \)

When \(K_1 = 1\), Bob wins when \(K_2 = 1\) because Alice takes the first card, and Bob takes the last card.

When \(K_1 = 2\), Alice will win if \(K_2 = 1\) because she can start by taking \(1\) card from the first deck. Bob then takes \(1\) card from either deck (each of which have only \(1\) card left), and Alice takes the last card. But if \(K_2 = 2\) then Bob can always win regardless of whether Alice starts by taking \(1\) card or \(2\) cards.

When \(K_1 = 3\), Bob can always win when \(K_2 = 1\). If Alice takes the single card from the second deck, Bob takes \(1\) card from the first deck and adds a new deck of size \(2\) to the table. We now have two decks of size \(2\), and it's Alice's turn. That's a losing state for Alice as we saw previously.

If Alice takes \(1\) card from the first deck and adds a new deck of size \(2\), we now have decks of size \([2, 1, 2]\). Bob will pick up the pile of size \(1\) and again we're in the same losing state for Alice. If Alice takes \(2\) cards from the first deck, we'll have decks of size \([1, 1, 2]\). Bob now takes the whole deck of size \(2\). Alice gets the next card and Bob gets the last card. Finally, if Alice takes all \(3\) cards from the first deck, we'll have decks of size \([1, 2]\). Bob can take just \(1\) card from the deck of size \(2\) to win.

In the second sample case:

\( \begin{array}{c|c|c|c|c}\underline{K_1}&\underline{A[K_1]}&\underline{B[K_1]}&\underline{C[K_1]}&\underline{K_2} \\ 1&1&1&0&1 \\ 2&1&1&0&2 \end{array} \)

In the third sample case:

\( \begin{array}{c|c|c|c|c}\underline{K_1}&\underline{A[K_1]}&\underline{B[K_1]}&\underline{C[K_1]}&\underline{K_2} \\ 1&1&1&0&1 \\ 2&1&2&0&2 \\ 3&2&3&1&2 \end{array} \)

In the fourth sample case:

\( \begin{array}{c|c|c|c|c}\underline{K_1}&\underline{A[K_1]}&\underline{B[K_1]}&\underline{C[K_1]}&\underline{K_2} \\ 1&1&1&0&1 \\ 2&1&2&0&2 \\ 3&3&3&2&3 \\ 4&1&4&0&4 \\ 5&5&5&4&3 \\ 6&1&6&0&6 \\ 7&6&7&4&3 \\ 8&1&8&0&8 \end{array} \)
