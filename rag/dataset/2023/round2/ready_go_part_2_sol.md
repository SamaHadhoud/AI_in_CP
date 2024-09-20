We can augment our solution to part 1 with a bit more information to solve part 2. Additionally, now that the bounds are larger, depth-first search solutions will need to be more careful about stack size. Breadth-first solutions won't have this problem.

We'll build up a table, \(\text{captured}(i, j)\), the number of white stones you can capture if you play a black stone at \((i, j)\). Initially every entry in this table is \(0\).

When you run your search to find all of the white stones in a given group, keep track of the total number of white stones in the group. Call this \(K\). If and only if this group turns out to have exactly \(1\) adjacent empty space at \((i, j)\), then increment \(\text{captured}(i, j)\) by \(K\), indicating that we'll capture these \(K\) stones if we play at \((i, j)\). If multiple white groups share the same single adjacent empty space, we'll be summing the sizes of all of those groups.

At the end, your answer is the maximum value in \(\text{captured}\). This solution still has a time complexity of \(\mathcal{O}(R * C\)).
