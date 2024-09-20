The problem is equivalent to finding the maximum \(x\) where you can make \(x\) groups with distinct sizes \(1, ..., x\), and all words that belong to a certain group pairwise \(K\)-rhyme with each other.

The first step is to simplify things by reversing all strings so that we're dealing with prefixes of length \(K\) instead of suffixes. Then, order queries by increasing \(K\). For equal \(K\), we sort queries based on [Mo’s algorithm](https://cp-algorithms.com/data_structures/sqrt_decomposition.html#mos-algorithm).

Note that there are \(\sum |W_i|\) possible prefixes across all the words. We can uniquely identify each prefix by a rolling hashing on each word, or by inserting each word into a trie and using the trie node ID or memory address. These are the values for which we are maintaining the frequency within Mo’s algorithm.

At each query being handled by Mo's algorithm, we should be careful to only process active words for the current size \(K\). The overall time complexity is \(\mathcal{O}((\sum |W_i|)*\sqrt N)\). In practice, our model solution solves the full test set in under \(10\) seconds.
