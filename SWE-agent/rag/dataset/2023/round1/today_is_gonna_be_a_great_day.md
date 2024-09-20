*This could possibly be the best day ever!
And the forecast says that tomorrow will likely be a billion and 6 times better.
So make every minute count, jump up, jump in, and seize the day,
And let's make sure that in every single possible way… today is gonna be a great day!*

There's \(N\) days of summer vacation, and the \(i\)th day initially has greatness \(A_i\). However, the days just keep getting better! In particular, there will be \(Q\) changes, the \(i\)th of which changes the greatness of every day from \(L_i\) to \(R_i\) to be \(1\,000\,000\,006\) times what they were before!

After each query, you'd like to know: which day's greatness modulo \(1\,000\,000\,007\) is the largest? If multiple days tie, we're interested in the earliest of them. For example, if there are \(N = 4\) days with greatnesses \([4, 1, 4, 2]\), then days \(1\) and \(3\) have the largest remainder, so our answer would be \(1\). To keep the output size small, please print only the sum of answers to all queries.

**Constraints**

\(1 \le T \le 45\)
\(1 \le N \le 1{,}000{,}004\)
\(1 \le A_i \le 1{,}000{,}000{,}006\)
\(1 \le Q \le 500{,}000\)
\(1 \le L_i \le R_i \le N\)

The sum of \(N\) across all test cases is at most \(3{,}000{,}000\).

**Input Format**

Input begins with a single integer \(T\), the number of test cases. For each case, there is first a line containing a single integer \(N\). Then, there is a line containing \(N\) space-separated integers \(A_1, ..., A_N\). Then, there is a line containing a single integer \(Q\). \(Q\) lines follow, the \(i\)th of which contains two space-separated integers \(L_i\) and \(R_i\).

**Output Format**

For the \(i\)th test case, print a single line containing "`Case #i: `" followed by a single integer, the sum of the greatest day(s) after each of the \(Q\) queries.

**Sample Explanation**

In the first test case, there are \(N=3\) days with initial greatnesses of \([1, 20, 30]\). The first query modifies all \(3\) days, transforming their greatnesses \([1\,000\,000\,006, 999\,999\,987, 999\,999\,977]\), where the maximum is on day \(1\). After the second query, day \(3\) will have the largest greatness of \(999\,999\,977\). The overall answer is the sum of the greatest days, which is \(1 + 3 = 4\).

