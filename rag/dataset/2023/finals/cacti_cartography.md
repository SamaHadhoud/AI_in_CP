Cactus Park is a famous **t**ourist attraction in Sandlandia. It holds \(N\) cactus plants, numbered from \(1\) to \(N\). The cacti are connected by \(M\) bidirectional trails, with trail \(i\) connecting cacti \(A_i\) and \(B_i\).

From any cactus, it's always possible to get to any other cactus by taking some sequence of trails. There may be cycles in the park, where a cycle is any sequence of trails that lead from a certain cactus back to itself. The park also has a special property that each trail belongs to at most one simple cycle. In graph theory terms, we can say that the Cactus Park forms a [cactus graph](https://en.wikipedia.org/wiki/Cactus_graph).

The park owners want to replace some number of cacti with information kiosks to help guide the tourists. Cutting down cactus \(i\) and building a kiosk there costs the park \(C_i\) dollars. The owners want to build enough kiosks so that the shortest path from every remaining cactus to the closest kiosk does not exceed \(K\) trails. Please help the owners determine the minimum total cost required to satisfy this requirement.

# Constraints

\(1 \le T \le 65\)
\(1 \le N \le 500\)
\(1 \le K \le \min(N, 50)\)
\(1 \le A_i, B_i \le N\)
\(A_i \ne B_i\)
\(1 \le C_i \le 10^9\)

Each unordered pair \((A_i, B_i)\) appears at most once in a given test case.

# Input Format

Input begins with a single integer \(T\), the number of test cases. For each case, there is first a line containing three integers \(N\), \(M\), and \(K\). Then, there is a line containing \(N\) integers \(C_{1..N}\). Then, \(M\) lines follow, the \(i\)th of which contains two integers \(A_i\) and \(B_i\).

# Output Format

For the \(i\)th case, output `"Case #i: "`, followed by a single integer, the minimum total cost in dollars to satisfy the park owners' requirement.

# Sample Explanation

The first case is depicted below. Replacing just cactus \(2\) would meet the requirement, but would cost \(\$10\). Instead we can replace cacti \(1\) and \(4\) for a total cost of \(\$8\).

{{PHOTO_ID:885100906409208|WIDTH:350}}

The second case is depicted below. One solution is to replace cacti \(1\) and \(4\) for a total cost of \(\$6\).

{{PHOTO_ID:3652000121792782|WIDTH:500}}

In the third case, all the cactuses are already within \(2\) trails of each other, so we just need a single kiosk anywhere. We should cut down the cheapest cactus, cactus \(1\).

{{PHOTO_ID:2643202679188960|WIDTH:350}}

In the fourth case, we can cut down cacti \(1\), \(3\), and \(6\) for a total cost of \(9 + 3 + 4 = \$16\)

{{PHOTO_ID:365613766120939|WIDTH:500}}
