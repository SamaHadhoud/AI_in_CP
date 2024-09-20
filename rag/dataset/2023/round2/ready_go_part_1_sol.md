A group of white stones can be captured if and only if there's exactly \(1\) empty space adjacent to the group, since playing a black stone in that space would leave the group with \(0\) adjacent empty spaces.

Loop through each space on the board, and when you encounter a white stone, use any search algorithm (a straightforward breadth-first search will suffice) to find all of the white stones that are part of the same group. Once you know which stones are in the group, you can check all \(4\) orthogonal neighbors of each white stone to see how many empty spaces are adjacent. Take care not to count the same empty space multiple times if it's adjacent to multiple white stones in the group (a set is a useful structure for this).

If you find any white group with exactly \(1\) adjacent empty space, the answer is "`YES`", otherwise the answer is "`NO`". This solution has a time complexity of \(\mathcal{O}(R * C\)).
