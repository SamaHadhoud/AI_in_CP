Patrick's big sister Sam is visiting for Halloween, and Sandy wants to impress the two Stars by dressing as an astronaut. Unfortunately her regular outfit already resembles an astronaut, so she'll have to go as a computer scientist instead. Sandy knows a lot about [double stars](https://en.wikipedia.org/wiki/Double_star) in astronomy, but will now have to learn about double stars in graph theory.

There is an unrooted tree with \(N\) nodes, and an edge between nodes \(i\) and \(P_i\) (for each \(i \ge 2\)). A *double star* is a tree centered at some edge \(u \leftrightarrow v\), where \(u\) and \(v\) each have \(x\) chains connected to them, each chain consisting of \(y\) nodes (\(x, y \ge 1\)). Alternatively, you can think of a double star as two identical star graphs (each with \(x\) chains of length \(y\)) connected by their centers. Thus, a distinct double star is specified by a tuple \((u, v, x, y)\), where \(u \lt v\). 

Please help Sandy count the number of distinct tuples \((u, v, x, y)\) for which the tree contains the double star described by that tuple as a subgraph.

# Constraints

\(1 \leq T \leq 110\)
\(2 \leq N \leq 1{,}000{,}000\)
\(1 \leq P_i \leq N\)

The sum of \(N\) across all test cases is at most \(8{,}000{,}000\).

# Input Format

Input begins with an integer \(T\), the number of test cases. Each case has two lines. The first contains the integer \(N\), and the second contains the \(N-1\) integers \(P_2\) through \(P_N\).

# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by the number of distinct double star tuples \((u, v, x, y)\), where \(u \lt v\).

# Sample Explanation
The first sample tree is depicted below. There are \(5\) distinct double star tuples: \((1, 2, 1, 1)\), \((1, 2, 1, 2)\), \((1, 2, 2, 1)\), \((1, 4, 1, 1)\) and \((2, 6, 1, 1)\).

{{PHOTO_ID:1573931960080150|WIDTH:350}}

The second sample tree is depicted below. There are \(5\) distinct double star tuples: \((1, 2, 1, 1)\), \((1, 2, 1, 2)\), \((1, 8, 1, 1)\), \((1, 8, 2, 1)\) and \((2, 7, 1, 1)\).

{{PHOTO_ID:1286569645378322|WIDTH:500}}

In the third sample tree, no double stars exist because arm lengths of \(0\) are not allowed \((y \ge 1 )\).
