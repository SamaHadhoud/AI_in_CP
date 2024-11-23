You currently weigh $W$ units, but you would ideally weigh $G < W$ units. On some days you have the mental fortitude to eat well and exercise. On other days, not so much. Every day, with equal likelihood, your weight will either increase or decrease by $1$ unit.

While your regimen may be lax, you do have one strict rule: if your weight has ever been $w$ units, then you will never in the future allow it to exceed $w + L$. On any day where gaining $1$ unit of weight would violate this constraint, you will guarantee that you lose $1$ unit of weight instead.

On average, how many days will it take you to reach a weight of $G$ units? The answer can be expressed as the ratio of two integers, $p/q$ in lowest terms. Output $p \times q^{-1}$ (mod $998{,}244{,}353$). 

($a^{-1}$ is the unique positive integer less than $998{,}244{,}353$ that satisfies $a \times a^{-1} \equiv 1$ (mod $998{,}244{,}353$)).

# Constraints
\(1 \leq T \leq 90\)
\(1 \leq G \lt W  \leq 10^{18}\)
\(1 \leq L  \leq 10^{18}\)

# Input Format
Input begins with an integer \(T\), the number of test cases. Each case is a single line with the integers $W$, $G$, and $L$.

# Output Format
For the \(i\)th test case, print "`Case #i:` " followed by the expected number of days it will take you to reach a weight of $G$ units, expressed as described above.

# Sample Explanation
In the first case, you weigh $201$ units and wish to reach a weight of $200$ units. You'll allow yourself to weigh $1$ beyond your lowest weight, so on the first day you either lose $1$ unit and stop, or you gain $1$ unit, and then force yourself to lose $1$ unit, returning back to $201$ units.

So there's a $1/2$ probability of finishing in $1$ day, a $1/4$ probability of $3$ days, a $1/8$ probability of $5$ days, and so on. The sum of this series converges to $3$ days on average.

In the fourth case you aren't allowed to ever gain weight, so you'll reach your goal in $77{,}665{,}544{,}332{,}211 - 11{,}223{,}344{,}556{,}677 = 66{,}442{,}199{,}775{,}534$ days. $66{,}442{,}199{,}775{,}534$ % $99{,}8244{,}353 = 53{,}884{,}207$, the final answer.

