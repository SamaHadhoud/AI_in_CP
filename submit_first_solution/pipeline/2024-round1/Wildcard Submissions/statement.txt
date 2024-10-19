A [trie](https://en.wikipedia.org/wiki/Trie) is a tree where every node represents a prefix of a string. When a string $s$ is inserted into a trie, up to $|s| + 1$ nodes are created if they don't already exist: 

* a node representing the empty string, `''`, which is always the root of the trie
* a node representing the first character of $s$, which is a child of the root
* a node representing the first two characters of $s$, which is a child of the previous node
* ... and so on

You have $N$ wildcard strings, $S_1$ through $S_N$, which each consist of uppercase Latin letters and question marks. A wildcard string represents the set of all strings that you can create by replacing each question mark with some uppercase Latin letter.

If you add every string represented by each of your wildcard strings to a single trie (initially empty), how many nodes will that trie have? Output this number modulo $998{,}244{,}353$.

# Constraints
\(1 \leq T \leq 105\)
\(1 \leq N \leq 25\)
\(1 \leq |S_i| \leq 100\)

All characters are uppercase Latin letters or question marks.

# Input Format
Input begins with an integer \(T\), the number of test cases. Each case begins with a line that contains the integer \(N\). Then \(N\) lines follow, the \(i\)th of which contains the string $S_i$.

# Output Format
For the \(i\)th test case, print "`Case #i:` " followed by the number of nodes in the resulting trie, modulo $998{,}244{,}353$.

# Sample Explanation
In the first case, we insert the string `META` into the trie, creating $5$ nodes: $[$ `''`, `M`, `ME`, `MET`, `META` $]$. Inserting `MATE` would create the $5$ nodes $[$ `''`, `M`, `MA`, `MAT`, `MATE` $]$, but those first $2$ nodes already exist, so $3$ new nodes are created. In the end we have $8$ nodes.

In the second case, we insert the strings $[$ `AB`, `BB`, `CB`, ... `ZB` $]$ into the trie, which creates the $57$ nodes $[$ `''`, `A`, `AB`, `B`, `BB`, ...,  `Z`, `ZB` $]$. When we add the string `AC`, that creates one new node, `AC`.

In the third case, we insert all combinations of $2$ letters, giving us $1$ node for the empty string, $26$ nodes for the single letters, and $26*26=676$ nodes for the pairs of letters, for a total of $703$ nodes.
