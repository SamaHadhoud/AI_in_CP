The key insight is that for the operation of multiplying a number \(x\) by \(1{,}000{,}000{,}006\) modulo \(1{,}000{,}000{,}007\), doing it twice will just get us \(x\) back again. Therefore, the \(i\)th day will only ever take on two greatness values: \(A_i\) and \((A_i * 1{,}000{,}000{,}006) \text{ mod } 1{,}000{,}000{,}007\). We can consider an update of \(L_i..R_i\) to be "flipping" every value in the range to their complementary value.

Let's build a [segment tree](https://en.wikipedia.org/wiki/Segment_tree) on the array. Each node is responsible for some subarray \(A_{l..r}\) and will maintain the following information:

* the maximum greatnesses value in that range (i.e. the current greatnesses)
* the day (index) of said maximum value
* the maximum of the flipped greatnesses in that range (i.e. what the greatnesses will become after an update)
* the day (index) of said flipped maximum

When merging two child nodes to a parent, we set the max initial value in the parent node to be the max of the children's initial values, breaking ties by lowest day index. Likewise for the flipped max and index.

Using the well-known lazy propagation technique, we can update ranges in time \(\mathcal{O}(\log N)\). For each node affected by an update, we can simply swap the first two fields with the last two. After each update, we can query the entire array (just access the root node) for the day of the current maximum greatness.

Across \(Q\) queries, we get an overall running time of \(\mathcal{O}(Q \log N)\).
