# Broadcasting in NumPy

Broadcasting is a set of rules that NumPy uses to let arrays with different shapes work together in arithmetic operations. When shapes don't match, NumPy aligns dimensions from the right (the trailing dimensions) and tries to stretch dimensions of size `1` so the arrays become compatible.

-   A dimension can broadcast if it matches or is `1`.
-   If two dimensions differ and neither is `1`, broadcasting fails.

## Why "align from the right"?

NumPy compares array shapes starting from the rightmost dimension, since those describe the element-level structure.

Example of right alignment:

    Array A shape:      (5, 1)
    Array B shape:          (5,)
                         --------
    Aligned shapes:    (5, 1)
                        (1, 5)

## Simple Example

``` python
import numpy as np

A = np.array([[10],
              [20],
              [30]])   # shape (3,1)

B = np.array([1, 2, 3])  # shape (3,)

# Broadcasting:
# A becomes (3,3) by repeating its single column
# B becomes (1,3) by repeating its single row
print(A + B)
```

Output:

    [[11 12 13]
     [21 22 23]
     [31 32 33]]
