Hacker Cup contest strategy often involves a metagame, where choosing which problems to work on might just be an important decision. On a Quest to become more Pro, you encounter an oracle promising to teach you the contest meta if you play her own Meta-game.

The oracle presents a peg board with \(2N\) moving dots. The initial \(y\)-positions of the dots are given as two arrays \(A_{1..N}\) and \(B_{1..N}\). Each second, simultaneously, \(A_1\) will move to the end of \(B\), while \(B_1\) will move to the end of \(A\) (with all elements shifting left accordingly).

You can connect the dots to form a *Meta-like logo* if all of the following are true:
* For the first half of both arrays, each dot in \(A\) is below the corresponding dot in \(B\).
* For the last half of both arrays, each dot in \(A\) is above the corresponding dot in \(B\).
* \(A\) equals the reverse of \(B\).

Formally:
* \(A_i < B_i\) for every \(i < (N+1)/2\)
* \(A_i > B_i\) for every \(i > (N+1)/2\)
* \(A_i = B_{N-i+1}\) for every \(i = 1..N\)

Note that if \(N\) is odd, the arrays' middle elements are not subject to the first two constraints.

The following is a visualization of a Meta-like logo (corresponding to the first sample case), with dots in \(A\) shown in red and dots in \(B\) shown in blue.

{{PHOTO_ID:359057163229199|WIDTH:500}}

You must answer the oracle: how many seconds must pass before the first time a Meta-like logo appears? If one never appears, output \(-1\).

# Constraints
\(1 \leq T \leq 135\)
\(2 \leq N \leq 2{,}000{,}000\)
\(0 \leq A_i, B_i \leq 1{,}000{,}000{,}000\)

The sum of \(N\) across all test cases is at most \(9{,}000{,}000\).


# Input Format

Input begins with an integer \(T\), the number of test cases. For each case, there is first a line containing a single integer \(N\). Then, there is a line containing integers \(A_1, ..., A_N\). Then, there is a line containing integers \(B_1, ..., B_N\).


# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by a single integer, the number of seconds that must pass before a Meta-like logo appears, or \(-1\) if that will never happen.


# Sample Explanation

The first test case is shown above. \(A\) and \(B\) already form a Meta-like logo, so the answer is 0.

The second case is not initially a Meta-like logo, for several reasons. One reason is that it is not symmetric. Specifically, the \([3, 3, 2, 3, 5 ,6]\) is not the reverse of \([4, 4, 6, 5, 3 ,2]\). After \(1\) second though, this case turns into the case above and is Meta-like.

The third and fourth cases will never turn into a Meta-like logo, no matter how many seconds we wait.

In the fifth case, after 6 seconds we see the first Meta-like logo. In this case \(A = [1, 1, 2, 2]\) and \(B = [2, 2, 1, 1]\).


