Let \(L\) be the number of leaves in the tree. A partition that optimally learns topics for all players might as well start each player's path from a leaf. Therefore there can be at most \(L\) paths to consider in a given partition.

Let \(F(s)\) be the frequency of topic \(s\) in the input. If a topic \(s\) is mutually-learnable, then it must be that \(F(s) \ge L\). This means there can be at most \(\mathcal{O}((\sum M_i) / L)\) candidate words. Assuming we have an \(\mathcal{O}(L)\) algorithm to verify that a candidate word is mutually-learnable, then we would have an overall running time of \(\mathcal{O}((\sum M_i) / L) * \mathcal{O}(L) = \mathcal{O}(\sum M_i)\), which is sufficient.

We can compress the tree by merging "lines", or sequences of nodes with only one child each. This can be done with a DFS, where if we reach a node \(u\) with \(1\) child, we repeatedly "reel up" the single-child chain (merging the topic lists into node \(u\)), Once the last single-child's children has become \(u\)'s children, we branch out from \(u\).

With this compression, we guarantee that each tree node has at least \(2\) children. This means that the numbers of nodes at each level from the bottom to the root will be limited by \(L\), \(\frac{L}{2}\), \(\frac{L}{4}\), …, \(1\). The total sum of this geometric series, as we should know, is \(\mathcal{O}(L)\).

To check each candidate word, we can do a DFS on the compressed tree, resolving the problem bottom-up. If we find the string in the root of the subtree, we pair it with any leaf needing it. However, if at any time a subtree has two leaves or more with no matches, that topic can’t be mutually-learned.
