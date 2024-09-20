*Bejeweled™* is a classic puzzle game where the player tries to match three-in-a-row in a 2D grid by swapping adjacent tile pairs. You may remember Hacker Cup's [spinoff](https://www.facebook.com/codingcompetitions/hacker-cup/2022/final-round/problems/C) by the name of *Isblinged™*. The all new sequel, *Isblinged 2*, is also played on a grid of \(R\) rows by \(C\) columns of tiles where the tile at \((i, j)\) is of an integer type \(G_{i,j}\).

At any time, the *score* of the grid is the number of subarrays of \(3\) equal tiles in either a single row or column (i.e. either a \(3 \times 1\) or \(1 \times 3\) submatrix). Note that different subarrays can overlap, and will each count toward the score. The score of the initial grid is guaranteed to be \(0\).

You will make exactly \(2\) moves, where each involves swapping a pair of adjacent tiles (either in the same row or column). What is the maximum score that can be achieved after the \(2\) moves?

# Constraints

\(1 \le T \le 100\)
\(1 \le R, C \le 1{,}000\)
\(R*C \ge 2\)
\(1 \le G_{i,j} \le 1{,}000{,}000\)

The sum of \(R * C\) across all test cases is at most \(4{,}000{,}000\).

# Input Format

Input begins with an integer \(T\), the number of test cases. For each case, there is first a line containing two space-separated integers, \(R\) and \(C\). Then, \(R\) lines follow, the \(i\)th of which contains \(C\) space-separated integers \(G_{i,1..C}\).

# Output Format

For the \(i\)th case, print a line containing `"Case #i: "` followed by a single integer, the maximum score.

# Sample Explanation

In the first case, one possible optimal ordered pair of swaps is depicted below:

{{PHOTO_ID:1050535959425825|WIDTH:750}}

The total score is \(5\) as there are \(3\) equal subarrays in the the first row and \(2\) in the second.

