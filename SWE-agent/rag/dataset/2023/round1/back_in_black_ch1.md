*This problem shares some similarities with C2, with key differences in bold.*

A friend who works at Metal Platforms Inc just lent you a curious puzzle, offering you tickets to a metal concert if you can solve it.

The puzzle consists of \(N\) buttons in a row, numbered from \(1\) to \(N\). The initial state of button \(i\) is white if \(S_i = 1\), or black if \(S_i = 0\). Pressing a button \(k\) toggles the state of itself as well as every \(k\)th button. Your friend challenges you to return the puzzle to him with all buttons back in black.

Life is hard enough without siblings pushing your buttons. Unfortunately, your brother has taken the puzzle and will push \(Q\) buttons sequentially, the \(i\)th being button \(B_i\). 

After your brother **has pushed all \(Q\) buttons**, you'd like to know the minimum number of button presses required to turn all the buttons black.

**Constraints**

\(1 \le T \le 70\)
\(1 \le N \le 4{,}000{,}000\)
\(1 \le Q \le 4{,}000{,}000\)

The sum of \(N\) and \(Q\) over all cases will be at most \(9{,}000{,}000\).

**Input Format**

Input begins with a single integer \(T\), the number of test cases. For each case, there is first a line containing a single integer \(N\). Then, there is a line containing a bitstring \(S\) of length \(N\). Then, there is a line containing a single integer \(Q\). \(Q\) lines follow, the \(i\)th of which contains a single integer \(B_i\).

**Output Format**

For the \(i\)th test case, output a single line containing `"Case #i: "` followed by a single integer, the number of button presses needed to turn all buttons black **after all \(Q\) button presses.**

**Sample Explanation**

The first sample case is depicted below. After your brother presses the first button, the state of the puzzle is \(0101\). The best strategy is to press the second button, turning all lights off.

{{PHOTO_ID:287687020772716|WIDTH:400}}

In the second case, the puzzle starts as \(0001\), and after each button press becomes \(0100\), \(0110\), \(0011\), and \(0010\) respectively. Pressing only the third button will return the puzzle to all zeros.
