For each vertex \(u\), and for each incident edge \(e\), we first have to calculate the longest distance to another vertex coming from this edge \(e\). It can be observed that for each vertex, there can be at most \(\mathcal{O}(\sqrt N)\) distinct distances.

We'll build a table \(d(u, v)\) storing the longest arm length starting with \(u \to v \to ...\), which can be computed in \(\mathcal{O}(N)\) using two DFS's. For each vertex \(v\), we have a list of pairs \((d, x)\) indicating that from vertex \(v\) there are \(x\) different edges incident to \(v\) in which the furthest distance you can reach is \(d\). We'll store this in a table \(f(u)\) holding a map of \((d, x)\), denoting that \(d\) is the greatest distance of an arm \(u \to v\) for \(x\) edges \(v\). \(f(u)\) can be computed using just the table \(d(u, v)\),

From here, the problem can be solved using the two-pointer technique. We'll want consider every edge \(u \leftrightarrow v\), and efficiently generate every \((x, y)\) that defines a double star tree with \(u \leftrightarrow v\) as its center. For each \((u, v, x, y)\), we iterate through \(f(u)\) and \(f(v)\) with two-pointers, with \(d'\) initially set to \(1\):
  - For the first \((d, x)\), take the sum of \(\min(\text{degree}(u) - 1, \text{degree}(v) - 1)\) over every \([d', d]\) (on those \(d\)'s, we can make a double star with that number of arms).
  - Remove the first and decrease \(x\) from \(\text{degree}(u)\) or \(\text{degree}(v)\), depending on what the pair belongs to.
  - Set \(d' = d + 1\).

The are \(\mathcal{O}(N)\) possible edges for the center of the star, each of which we'll examine \(\mathcal{O}(\sqrt N)\) distinct distances using the two-pointers. The overall time complexity is \(\mathcal{O}(N \sqrt N)\).
