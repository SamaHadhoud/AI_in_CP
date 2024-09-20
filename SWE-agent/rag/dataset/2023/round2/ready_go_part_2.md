_This problem shares some similarities with A1, with key differences in bold._

_Atari 2600? More like Atari 2600 BCE!_

The classic board game _Go_ is a two-player game played on an \(R \times C\) board. One player places white stones while the other places black stones. On a player's turn, they may place a stone in any empty space. A curiosity of Go is that stones are placed on the intersections of grid lines rather than between the lines, so an in-progress \(5 \times 5\) game looks like this:

{{PHOTO_ID:142163805651048|WIDTH:180}}

An orthogonally contiguous set of stones of the same color is called a *group*. A group of stones is captured (and removed from the board) once no stones in the group has an adjacent empty space.

You're playing as Black and it's your turn. Given a valid board (i.e. no groups have \(0\) adjacent empty spaces), **what’s the maximum number of white stones you can capture** with a single black stone?

Here are some examples of captures. If a black stone is placed at the point marked with a triangle, a single white stone will be captured:

{{PHOTO_ID:1327337367897977|WIDTH:300}}

Here, Black can capture a group of \(3\) white stones. Note that this move is valid even though the new black stone has no adjacent empty spaces at the moment it's placed:

{{PHOTO_ID:311797128262237|WIDTH:240}}

Black can even capture multiple groups at once. Here, Black captures a group of \(2\) stones and a group of \(3\) stones:

{{PHOTO_ID:975143063563862|WIDTH:400}}

The Go board is represented as a character array \(A\) where \(A_{i, j}\) is one of:
* `B` for a black stone
* `W` for a white stone
* `.` for an empty space


# Constraints

\(1 \leq T \leq 150\)
\(1 \leq R, C \leq \textbf{3{,}000}\)
\(A_{i, j} \in \{\)'`.`', '`B`', '`W`'\(\}\)


# Input Format

Input begins with an integer \(T\), the number of test cases. Each case begins with a line containing two integers \(R\) and \(C\). Then, \(R\) lines follow, the \(i\)th of which contains \(C\) characters \(A_{i, 1}\) through \(A_{i,C}\).


# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by a single integer, the maximum number of white stones you can capture on your turn.


# Sample Explanation

In the first case, Black can capture 3 white stones by playing in the bottom-right corner.

In the second case, there are no white stones that can be captured.

In the third case, Black can capture both white groups at once, for a total of 6 + 3 = 9 white stones.

In the fourth case, there are 6 different white stones that can be captured, but Black can capture at most 4 of them at once (by playing in the center of the board).

