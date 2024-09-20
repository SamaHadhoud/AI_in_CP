Alice and Bob are servers at *Nim Sum Dim Sum*, a bustling dumpling restaurant. For a staff meal, the manager has generously provided \(N\) plates of dumplings in a row, numbered from \(1\) to \(N\). Initially, plate \(i\) has \(A_i\) dumplings. In classic fashion, the duo has decided to play a game.

Alice and Bob will take turns eating dumplings from the plates. On a given turn, a player must pick a plate adjacent to the last picked plate by the other player, and eat any positive number of dumplings from that plate. The first player who cannot do so on their turn loses. Alice goes first, and can choose any starting plate to eat from.

For example, suppose there are three plates holding \(4\), \(1\) and \(2\) dumplings respectively. On the first turn, Alice can eat \(3\) dumplings from the first plate. Bob must then eat the dumpling from the middle plate. Alice can respond by eating one dumpling from the third plate. Bob must then eat from plate \(2\), but since it’s empty now, he loses.

Assuming both players play optimally, how many starting moves can Alice make so that she wins? Two starting moves are considered different if Alice eats from different plates, or eats a different number of dumplings.

# Constraints

\(1 \le T \le 220\)
\(2 \le N \le 800{,}000\)
\(0 \le A_i \lt 2^{25}\)

The sum of \(N\) across all test cases is at most \(4{,}000{,}000\).

# Input Format

Input begins with an integer \(T\), the number of cases. Each case will begin with a single integer \(N\) followed by the \(N\) integers \(A_1, ..., A_N\) on the next line.

# Output Format

For the \(i\)th case, output `"Case #i: "` followed by a single integer, the number of winning starting moves Alice has.

# Sample Explanation

In the first case, Alice can start by taking any number of dumplings from either the first or third plate. Bob will then have to take the solitary dumpling on the middle plate, and Alice can win by taking all the dumplings from the plate she didn't start with. This gives Alice 6 different winning starting moves.

In the second case, Alice cannot win because she takes one dumpling, Bob takes the other, and then Alice has no move to make.

In the third case, Alice's winning moves are to take \(1\) or \(2\) dumplings from the right-hand plate.

In the fourth case, Bob can always force a win.

