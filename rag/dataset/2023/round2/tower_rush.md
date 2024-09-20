Alice and Bob are playing *Tower Rush,* a two-phase game involving \(N\) types of blocks, numbered from \(1\) to \(N\). Blocks of type \(i\) have height \(H_i\), and no two types of blocks have the same height.

Phase 1 of the game consists of \(K \ge 2\) alternating turns, with Alice going first. On each player's turn, they will choose a block type that has not yet been chosen. Note that if \(K\) is odd, this phase would end with Alice having chosen one more type of block than Bob. From this point on, players have access to infinitely many blocks of each of the types they chose.

Phase 2 consists of an indefinite number of alternating turns, with Bob going first. Each player starts with a tower of height \(0\). On each player's turn, they may pick a block of any type \(i\) available to them, and extend their tower height by \(H_i\). They may also choose to skip their turn, leaving their tower unchanged.

In a break from tradition, Alice and Bob want to work *together* to see if it's possible for Alice to build a tower that's exactly \(D\) units taller than Bob's. In how many different ways can phase 1 be played out such that it will be possible for Alice to get her tower to be exactly \(D\) units taller than Bob's in phase 2? Two ways are considered different if there exists an \(i\) such that different block types are chosen on turn \(i\) between the two ways.

As this value may be large, output it modulo \(1{,}000{,}000{,}007\).


# Constraints

\(1 \leq T \leq 24\)
\(2 \leq N \leq 1{,}000{,}000\)
\(2 \leq K \leq \min(N, 20)\)
\(1 \leq D \leq 1{,}000{,}000\)
\(1 \leq H_i \leq 1{,}000{,}000\)

The sum of \(N\) across all test cases is at most \(7{,}500{,}000\).


# Input Format

Input begins with an integer \(T\), the number of test cases. For each case, the first line contains the integers \(N\), \(K\), and \(D\). Then, there is a line containing the \(N\) integers \(H_1\), ..., \(H_N\).


# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by a single integer, the number of unique ways to accomplish Bob's goal, modulo \(1{,}000{,}000{,}007\).


# Sample Explanation

In the first sample case, there are \(54\) valid choices of blocks in phase 1. One such example is \(\{H_2 = 4,\, H_3 = 5\}\) for Alice and \(\{H_4 = 7\}\) for Bob. Four turns in phase 2 can go as follows:
* Bob adds \(H_4 = 7\). His tower now has height \(7\).
* Alice adds \(H_3 = 5\). Her tower now has height \(5\).
* Bob chooses not to add a block. His tower still has height \(7\).
* Alice adds \(H_3 = 5\). Her tower now has height \(10\).

Alice’s tower is now exactly \(3\) units taller than Bob’s, so this is a valid choice.

An example of an invalid choice in the first test case would be \(\{H_1 = 2,\, H_2=4\}\) for Alice and \(\{H_5 = 8\}\) for Bob, since no matter how they build their towers, it is impossible for the difference in the height of their towers to ever be \(3\).
