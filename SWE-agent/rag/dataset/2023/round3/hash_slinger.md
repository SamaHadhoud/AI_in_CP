Spongebob wants to go out trick-or-treating on Halloween, but has to work a graveyard shift at the Krusty Krab. Luckily, his ghostly fry cook friend, the hash-slinging slasher, is here to cover.

At the start of the shift, there are \(N\) patties on the grill (numbered from \(1\) to \(N\)), the \(i\)th of which weighs \(A_i\) grams. For each order that comes in, the slasher removes from the grill a non-empty sequence of patties at contiguous indices \(i..j\) (patties \(i..j\) must all be on the grill at that moment). The *deliciousness* of the order is defined as \((A_i + ... + A_j)\), modulo \(M\) because too much meat can be overpowering! Note that if a patty is removed, a future order's range cannot span across that empty spot. Also, there may be patties left over at the end.

As proof of his hard work at the end of the shift, the slasher will compute a hash by taking the [bitwise XOR](https://en.wikipedia.org/wiki/Bitwise_operation) of the deliciousnesses of all the orders. How many distinct hashes are possible across all valid sequences of \(0\) or more orders?


# Constraints

\(1 \leq T \leq 125\)
\(0 \leq A_i \leq 1{,}000{,}000{,}000\)
\(1 \leq N \leq 9{,}000\)
\(1 \leq M \leq 5{,}000\)

The sum of \(N + M\) across all test cases is at most \(200{,}000\).


# Input Format

Input begins with an integer \(T\), the number of test cases. Each case begins with a line containing the two integers \(N\) and \(M\). Then, there is a single line containing \(N\) integers, \(A_1, ..., A_N\).


# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by the number of different possible hashes.


# Sample Explanation

In the first sample case, there are \(3\) patties with \(A = [2, 2, 0]\) and \(M = 10\). The possible valid order sequences (patties removed) and hashes are:

- No patties removed \(\to\) hash \(0\)
- Patties \(1..1\) \(\to\) hash \(2\)
- Patties \(2..2\) \(\to\) hash \(2\)
- Patties \(3..3\) \(\to\) hash \(0\)
- Patties \(1..2\) \(\to\) hash \(4\)
- Patties \(2..3\) \(\to\) hash \(2\)
- Patties \(1..3\) \(\to\) hash \(4\)
- Patties \(1..1\) and patties \(2..2\) \(\to\) hash \(0\)
- Patties \(1..1\) and patties \(3..3\) \(\to\) hash \(2\)
- Patties \(2..2\) and patties \(3..3\) \(\to\) hash \(2\)
- Patties \(1..1\) and patties \(2..3\) \(\to\) hash \(0\)
- Patties \(1..2\) and patties \(3..3\) \(\to\) hash \(4\)
- Patties \(1..1\), patties \(2..2\), and patties \(3..3\) \(\to\) hash \(0\)

There are \(3\) distinct hashes the slasher can make: \(0\), \(2\), and \(4\).

In the second case, the slasher can make hashes \(0\) (for example, with no orders) and \(1\) (for example, with \(5\) orders, each including one of the patties).



