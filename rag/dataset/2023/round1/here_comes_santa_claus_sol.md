For a given assignment, Santa will always walk from the leftmost toy to the rightmost toy. It's in our interest to get these two meeting points as far as possible. This is equivalent to independently minimizing the leftmost meeting point, while maximing the rightmost meeting point.

For all elves assigned to a toy, the meeting point will always be the average of the leftmost and rightmost elf. To minimize the leftmost meeting point, it's intuitive that grouping only the \(2\) leftmost elves will get the average as small as possible. Likewise, we group the \(2\) rightmost elves.

We can sort the elves by ascending \(x\)-coordinates \(X_1' < ... < X_N'\), find the left meeting point as \((X_2' - X_1')/2\) and the right meeting point as \((X_{N}' - X_{N-1}')/2\). The answer (i.e. the distance of Santa's walk) will be the difference between those two meeting points.

The only special case occurs if there are \(N = 5\) elves. If we did the above, we would be left with a single elf in the middle who isn't allowed to work on a toy alone. We must instead include the middle elf in one of the other two groups. When \(N \ge 6\), there are at least \(2\) middle elves in our main strategy, so they won't pose an issue.

