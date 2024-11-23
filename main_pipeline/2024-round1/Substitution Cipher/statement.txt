You’ve probably seen the most classic substitution cipher, where `1` = `A`, `2` = `B`, and so on. Under this scheme, a string like `META` would be encoded as `13 5 20 1`.

You’ve just received an encoded string of digits, $E$, but unfortunately all of the spaces have been removed and some (possibly $0$) of the digits have become corrupted (represented as question marks). 

You must first “uncorrupt” the string by replacing each instance of `?` with a digit and then splitting the uncorrupted string into a series of numbers that are all between $1$ and $26$ inclusive. For example, the string `?35?01` can be uncorrupted to `135201`. That uncorrupted string can be split into `13 5 20 1` as above, but also `1 3 5 20 1`, which decodes to `ACETA`. Note that leading zeros are not valid, so `13 5 2 01` is not a valid way to split the uncorrupted string, so this uncorrupted string cannot be decoded into `MEBA`.

Given a corrupted encoded string, what uncorrupted encoded string has the largest number of possible strings that it could decode into, and how many decoded strings is that?

As there may be multiple uncorrupted encoded strings that share the same maximum answer, output the lexicographically \(K\)th largest among them. Output the number of decoded strings that this uncorrupted encoded string can decode into modulo $998{,}244{,}353$.

# Constraints
\(1 \leq T \leq 85\)
\(1 \leq |E| \leq 100{,}000\)
\(1 \leq K \leq 1{,}000{,}000\)

* Every character in $E$ is either a digit or a question mark.
* \(K\) will never be larger than the number of uncorrupted strings that have the most possible decoded strings.
* There will always be some uncorrupted version of $E$ that can be properly decoded.

The sum of \(|E|\) across all test cases is at most $400{,}000$.

# Input Format
Input begins with an integer \(T\), the number of test cases. Each case is a single line that contains the string \(E\) followed by the integer \(K\).

# Output Format
For the \(i\)th test case, print "`Case #i:` " followed by the lexicographically \(K\)th-largest uncorrupted encoded string amongst all possible uncorrupted encoded strings that can be decoded into the maximum number of decoded strings, followed by what that maximum is, modulo $998{,}244{,}353$.

# Sample Explanation
In the first case, possible uncorrupted strings include `282`, but strings like that can only decode to $1$ string, whereas other strings like `212` can be decoded into $3$ different strings, the maximum possible. The uncorrupted strings that can be decoded into $3$ different strings are [`222`, `212`, `122`, `112`], so `122` is the lexicographically third-largest.

The second case is described above. The string has no corrupted digits and can be decoded into $2$ different strings.

In the third case, to maximize the number of possible decodings for a given uncorrupted string, which is $2$, the two options are `235` and `135`.

In the fourth case, the two options are `120` and `110`, each of which decode to just $1$ string.

In the fifth case, there are no uncorrupted digits and the encoded string can be decoded into \(5\) different strings: `KV`, `KBB`, `ANB`, `AAV`, `AABB`.