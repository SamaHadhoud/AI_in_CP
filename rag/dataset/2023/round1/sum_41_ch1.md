*This problem shares some similarities with B2, with key differences in bold.*

Given a positive integer \(P\), please find an array of at most \(100\) positive integers which have a sum of \(41\) and a product of \(P\), or output \(-1\) if no such array exists.

If multiple such arrays exist, **you may output any one of them**.

# Constraints
\(1 \leq T \leq 965\)
\(1 \leq P \leq 10^9\)

# Input Format

Input begins with an integer \(T\), the number of test cases. For each case, there is one line containing a single integer \(P\).

# Output Format

For the \(i\)th test case, if there is no such array, print "`Case #i: -1`". Otherwise, print "`Case #i:` " followed by the integer \(N\), the size of your array, followed by the array itself as \(N\) more space-separated positive integers.

# Sample Explanation
In the first sample, we must find an array with product \(2023\), and sum \(41\). One possible answer is \([7, 17, 17]\).

