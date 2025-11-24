#  Understanding Cross-Entropy Loss and Softmax:  
## Why Only the Correct Class Appears — Yet Every Class Learns

Cross-entropy is the default loss function for classification in modern deep learning.  
But one part often confuses learners:

> Why does cross-entropy only include the probability of the correct class,  
> yet the network still updates all class logits?

This article provides a clean, rigorous, and intuitive explanation.



# 1. What Cross-Entropy Actually Measures

For a single sample with true class $y$ and predicted probabilities $p_i$ from softmax:

$$
L = -\log(p_{\text{correct}}) = -\log(p_y)
$$

It seems cross-entropy cares only about the correct class.  
And that part is true.

But that’s not the full story.

# 2. Softmax Creates Competition Among Classes

Softmax converts logits $z_i$ into probabilities:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Softmax ensures:

$$
\sum_i p_i = 1
$$

This couples all classes together.  
Increasing the logit of any incorrect class automatically reduces the probability of the correct class.

Thus, even though the loss formula only includes $p_y$,  
changing any logit $z_i$ changes $p_y$.

This is why incorrect classes still influence the loss.


# 3. The Real Reason Every Class Learns: The Gradient

The most important fact in cross-entropy + softmax is this gradient:

$$
\frac{\partial L}{\partial z_i} = p_i - Y_i
$$

Where:

- $Y_i = 1$ for the correct class
- $Y_i = 0$ for all incorrect classes

This single equation explains everything:

- For the correct class $i=y$:

$$
\frac{\partial L}{\partial z_y} = p_y - 1
$$

- For every incorrect class:

$$
\frac{\partial L}{\partial z_i} = p_i
$$

### ✔ All incorrect classes get gradients proportional to their predicted probabilities  
### ✔ The correct class gets pushed upward  
### ✔ Incorrect classes get pushed downward

This makes the “competition” between classes mathematically explicit.

Cross-entropy doesn't have to include incorrect probabilities explicitly —  
the gradient already penalizes them.
-

#  4. Why Frameworks Don’t Explicitly Apply Softmax

In PyTorch / TensorFlow, the loss takes logits, not probabilities:

```python
CrossEntropyLoss(logits, labels)
