*Is this a real verse? Is this just fantasy?
Caught in a mixtape, no escape from this melody.
Open your ears, listen to the beats and see.
I'm just a rapper, I need no sympathy,
Because it's easy come, easy go, little high, little low,
Any way the bars flow doesn't really matter to me, to me.*

As an up-and-coming rapper, you're trying to distinguish yourself by writing a song with the most bohemian rhyme scheme!

Your vocabulary consists of \(N\) words, \(W_1\), \(…\), \(W_N\) made up of only lowercase letters. We consider two words to *\(K\)-rhyme* with each other if they have the same suffix of length \(K\). For example, “mixtape” 3-rhymes with “escape”. If two words \(K\)-rhyme for \(K > 1\), they also \((K-1)\)-rhyme.

You are faced with \(Q\) bohemian rap-queries, the \(i\)th of which specifies \((A_i, B_i, K_i)\). You’d like to assign each word \(W_{A_i}\), \(…\), \(W_{B_i}\) to at most one group such that:

* the number of words in each group is unique and nonzero,
* groups only contain words with length at least \(K_i\),
* within each group, all pairs of words \(K_i\)-rhyme with each other, and
* no two words from different groups \(K_i\)-rhyme.

The answer to each query is the maximum possible number of nonempty groups. For each test case, output the sum of the answers over all queries. 

**Constraints**

\(1 \le T \le 20\)
\(1 \le N \le 600{,}000\)
\(\sum |W_i| \le 3{,}000{,}000\)
\(1 \le Q \le 400{,}000\)
\(1 \le A_i \le B_i \le N\)
\(1 \le K_i \le 1{,}000{,}000\)
All words consist of only letters \(\{\texttt{`a'}, ..., \texttt{`z'}\}\)
All words within a test case are distinct.

The sum of \(N\) across all test cases is at most \(2{,}000{,}000\).
The sum of \(Q\) across all test cases is at most \(1{,}000{,}000\).
The sum of word lengths across all test cases is at most \(10{,}000{,}000\).

**Input Format**

Input begins with an integer \(T\), the number of test cases. For each case, there is first a line containing a single integer \(N\). \(N\) lines follow, the \(i\)th of which contains the string \(W_i\). Then, there is a line containing a single integer \(Q\). \(Q\) lines follow, the \(i\)th of which contains \(3\) space-separated integers \(A_i\), \(B_i\), and \(K_i\).

**Output Format**

For the \(i\)th test case, print "`Case #i:` " followed by a single integer, the sum of answers over all queries.

**Sample Explanation**

For the first query in the first test case, we are allowed to choose from all \(W_1, ..., W_8\) words to form \(3\) groups where words of the same group \(1\)-rhyme with each other. One possible choice of the groups is: \(\{\)"low"\(\}\), \(\{\)"sympathy", "fantasy", "melody"\(\}\), \(\{\)"mixtape", "escape"\(\}\) with "come" and "high" going unused. Therefore the answer to the first query is \(3\). The answers to the remaining queries are \(1\), \(2\) and \(2\), with the overall sum of answers for this test case equal to \(8\).

In the second test case, the answers to the \(3\) queries are \([2, 1, 0]\) with the sum being \(3\).

