First note that for a given starting location with the optimal \(Q\) for that location, one viable solution is to always greedily take over the lowest value adjacent city to your current set of cities.

To see this we can use an exchange argument. Consider a situation we have power \(P\) and are adjacent to some set of cities \(S\). If we do not take the city of minimum value first, consider the alternative where we took the minimum first instead. This would always be possible, as if we can take any city we can take the one of minimum cost. Also, as we only increase to size of our reclaim set, this cannot make any future takeovers harder. Thus, given any strategy we can always improve our takeover strategy by greedily taking the smallest adjacent city, showing that with the minimum \(Q\) the greedy strategy will always work.

Another modification we can make is to instead of paying up-front for \(Q\) we can instead take any city, but if the value of the city \(i\) exceeds our current power \(P\) we pay a penalty of \(P_i - P\) instead. However, note that these penalties don't add; instead we only need to pay the maximum of these penalties.

This inspires a naive \(\mathcal{O}(N^2\log N)\) algorithm, where for each starting node we greedily add the lowest value neighbor and compute the penalties we take. The answer for each node is the maximum such penalty.

However we can improve this as follows:

We keep track of a Union Find data structure containing all cities that have been combined, along with the total power of that component. We iterate over cities \(i\) in increasing order of \(P_i\). For each iterated \(i\) find all adjacent nodes of smaller value (ie. are earlier in the order), calculate the penalty for that component to reclaim city \(i\), and store it in the Union Find node for that component. Once this is done, union \(i\) with all of the adjacent components we just saw.

We can show this cost to reclaim city \(j\) is the least possible of the needed extra power that we will need to ‘escape’ from the smaller component, and thus a lower bound for the needed \(Q\) for all nodes in the component. To see this, note that all adjacent nodes to the current reclaimed set \(S\) must have value at least \(P_j\) (otherwise they would have been added to the component before \(j\)). This means in any order of claiming cities the first time we claim a node with value at least \(P_j\), our set of cities is a subset of \(S\) and has sum at most that of \(S\). Thus any possible excess value needed when this happens is at least \(P_j - \sum_{i \in S}  P_i\), which is the value from before.

Once we have iterated over all of the nodes, note that the answer for a node is just the maximum of all edges above it in the Union Find tree. We have shown it is a lower bound for the added power above, and to show that this power is sufficient we just take the nodes in the order that were connected to the starting node in the Union Find.

Thus, the answer can be calculated for each node by iterating the tree in topological order and for each node computing the answer the maximum of the edge above it and the answer of its parent node.

The overall complexity is \(\mathcal{O}(N\log N)\), corresponding to both sorting the city values as well as possibly for the Union Find itself (depending on the details of how it was implemented).

