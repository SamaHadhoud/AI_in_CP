*…right down Santa Claus Lane!*

Santa Claus Lane is home to \(N\) elves, the \(i\)th of whom lives \(X_i\) meters from the start of the lane. As Santa's little helper, you're tasked to assign elves to work on toys for good little girls and boys. Specifically, you must assign each elf to work on some toy so that at least \(2\) toys get worked on, and no elf works on a toy alone.

All elves working on the same toy will meet at the point which minimizes the farthest distance that any of them would need to walk. Formally, if the elves assigned to a given toy live at \(X_1\), \(X_2\), \(…\), then they will meet at point \(P\) such that \(\max(|X_1 - P|\), \(|X_2 - P|\), \(…\)\()\) is as small as possible.

For instance, the first sample case is depicted below:

{{PHOTO_ID:838516607729325|WIDTH:700}}

Santa is supervising, and you reckon he could use some exercise. Among all valid assignments of elves to toys, what's the farthest Santa would need to walk to visit all meeting points? Santa may start and end anywhere, but will try to walk as little as possible after seeing your assignments.

# Constraints

\(1 \leq T \leq 20\)
\(4 \leq N \leq 10^5 \)
\(1 \leq X_i \leq 10^9 \)

# Input Format

Input begins with an integer \(T\), the number of test cases. Each case begins with one line containing the integer \(N\), followed by a second line containing the the \(N\) integers \(X_1 ... X_N\).

# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by a single real number, the farthest distance Santa would need to walk in meters. Your answer will be considered correct if it differs from the jury's answer by at most \(10^{-6}\) meters or at most \(0.0001\)%, whichever is larger.

# Sample Explanation

In the first sample case, elves living at \(1 \,\text{m}\) and \(3 \,\text{m}\) will work on a toy. Elves living at \(7 \,\text{m}\), \(12 \,\text{m}\), and \(13 \,\text{m}\) will work on another toy, and elves at \(17 \,\text{m}\) and \(23 \,\text{m}\) will work on the third toy. The toys will be made at \(2 \,\text{m}\), \(10 \,\text{m}\) and \(20 \,\text{m}\) respectively. Santa would need to walk at least \(18 \,\text{m}\) in total to visit every meeting point.

The second sample case is depicted below. No elf is allowed to work alone and we must make two toys. One optimal way of doing this is to have the leftmost \(3\) elves work on one toy at \(2 \,\text{m}\), and the rest work on the other toy at \(4.5 \,\text{m}\). This would maximize the distance Santa would have to walk among all valid elf assignments.

{{PHOTO_ID:163905706775126|WIDTH:400}}

In the third case, the two toys will be made at \(55 \,\text{m}\) and \(5500 \,\text{m}\).

