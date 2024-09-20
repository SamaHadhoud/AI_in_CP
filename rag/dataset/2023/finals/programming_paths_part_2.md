_The only difference between chapters 1 and 2 is the maximum allowed grid size, given in bold below._

A *Drizzle* program is a 2D grid of the following four types of cells:
 - '`@`' (start) \(-\) there is exactly one start cell in the entire grid
 - '`#`' (wall)
 - '`.`' (space)
 - '`*`' (instruction)

The program uses two registers \(A\) and \(B\) (both initially \(0\)), and executes as follows:

1. Compute the minimum distance from the start to each instruction cell using orthogonal movements, without going outside of the grid or passing through any wall cells. Instruction cells that cannot be reached are ignored. 
2. In increasing order, for each unique distance \(D\) such that there’s at least one instruction cell that’s at distance \(D\) from the start cell:
   2a. Count the number of shortest paths, \(P\), to all instruction cells of distance \(D\).
   2b. Look up the instruction corresponding to \((P \text{ mod } 2, D \text{ mod } 2)\) in the table below and modify one of the registers accordingly.
3. At the end, the value in register \(A\) is outputted.

```
┌─────────────┬─────────────┬─────────────┐
│             │ D mod 2 = 0 │ D mod 2 = 1 │
├─────────────┼─────────────┼─────────────┤
│ P mod 2 = 0 │ A := A + 1  │ A := A - 1  │
│ P mod 2 = 1 │ B := B + A  │ A := B      │
└─────────────┴─────────────┴─────────────┘
```

For a given value \(K\), output any Drizzle program that outputs \(K\) when executed, with the restriction that **the program must fit on a \(\mathbf{10}\) × \(\mathbf{10}\) grid**.


# Constraints

\(1 \le T \le 2{,}000\)
\(0 \le K \le 10{,}000\)


# Input Format

Input begins with an integer \(T\), the number of test cases. For each case, there is a line containing the single integer \(K\).


# Output Format

For the \(i\)th case, output "`Case #i: `" followed by two integers \(R\) and \(C\), the number of rows and columns in your program, respectively. Then output your program. It must be exactly \(R\) lines long, with each line containing exactly \(C\) characters.


# Sample Explanation

Here are the instructions executed for each of the sample programs. Note that many other programs would be accepted for any for these cases.

In the first case, there is a single instruction. There are \(2\) shortest paths of length \(2\) to that instruction, so \(P = 2\) and \(D = 2\). That means we perform \(A := A + 1\). There are no more instructions, so the program ends and outputs \(1\).

In the second case, there are three instruction cells. Each of them are an even distance from the start, and each have an even number of shortest paths leading to them, so each represents \(A := A + 1\):

1) \(2\) paths of length \(2\) \(\;(A := A + 1 = 1)\)
2) \(4\) paths of length \(6\) \(\;(A := A + 1 = 2)\)
3) \(4\) paths of length \(12\) \(\;(A := A + 1 = 3)\)

In the third case, there are eight instruction cells, but some of them are at the same distance as each other. In particular, there are two instruction cells at distance \(2\), and three instruction cells at distance \(10\). There's a single shortest path to each of the cells at distance \(2\), so in total there are \(2\) shortest paths to instructions at distance \(2\). One of the cells at distance \(10\) has a unique shortest path, and the other has two shortest paths, so in total there are \(3\) shortest paths to instructions at distance \(10\).

1) \(2\) paths of length \(2\) \(\;(A := A + 1 = 1)\)
2) \(6\) paths of length \(4\) \(\;(A := A + 1 = 2)\)
3) \(1\) path of length \(6\) \(\;(B := B + A = 2)\)
4) \(1\) path of length \(8\) \(\;(B := B + A = 4)\)
5) \(3\) paths of length \(10\) \(\;(B := B + A = 6)\)
6) \(3\) paths of length \(11\) \(\;(A := B = 6)\)
