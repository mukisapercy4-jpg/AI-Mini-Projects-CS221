# Assignment 2: Sentiment Classification Theory

**Author:** Ssewajje Jolly Hilton

## Problem 1: Building Intuition

**Part a: SGD by Hand**
We run stochastic gradient descent (SGD) on the hinge loss. Let the feature mapping \(\phi(x)\) return word counts and let the initial weight vector be \(\mathbf{w} = [0, 0, 0, 0, 0, 0]\) for the tokens ("pretty", "good", "bad", "plot", "not", "scenery"). We use a learning rate \(\eta = 0.1\).

1. **"not good" (y = -1):** Score = 0. Margin = 0 < 1. Update: \(\mathbf{w} \leftarrow \mathbf{w} + 0.1(-1)\phi\). 
   \(\mathbf{w} = [0, -0.1, 0, 0, -0.1, 0]\)
2. **"pretty bad" (y = -1):** Score = 0. Margin = 0 < 1. Update: \(\mathbf{w} \leftarrow \mathbf{w} + 0.1(-1)\phi\).
   \(\mathbf{w} = [-0.1, -0.1, -0.1, 0, -0.1, 0]\)
3. **"good plot" (y = +1):** Score = -0.1. Margin = -0.1 < 1. Update: \(\mathbf{w} \leftarrow \mathbf{w} + 0.1(+1)\phi\).
   \(\mathbf{w} = [-0.1, 0.0, -0.1, 0.1, -0.1, 0]\)
4. **"pretty scenery" (y = +1):** Score = -0.1. Margin = -0.1 < 1. Update: \(\mathbf{w} \leftarrow \mathbf{w} + 0.1(+1)\phi\).
   \(\mathbf{w} = [0.0, 0.0, -0.1, 0.1, -0.1, 0.1]\)

**Final Weight Vector:** `[0.0, 0.0, -0.1, 0.1, -0.1, 0.1]`

**Part b: Linear Separability Proof**
Let the weights for "bad", "good", and "not" be \(w_b, w_g, w_n\). For zero error, the classifier must satisfy:
1. "bad" (-1): \(w_b < 0\)
2. "good" (+1): \(w_g > 0\)
3. "not bad" (+1): \(w_n + w_b > 0 \implies w_n > -w_b > 0\)
4. "not good" (-1): \(w_n + w_g < 0 \implies w_n < -w_g < 0\)

We cannot have \(w_n > 0\) and \(w_n < 0\) simultaneously. Thus, no linear classifier can perfectly separate this dataset.

**Additional Feature:** Add a bigram feature for **"not_bad"** and **"not_good"**. This allows the model to learn a positive weight for "not_bad" and a negative weight for "not_good" independent of the individual word weights.

## Problem 2: Predicting Movie Ratings

**Part a: Squared Loss**
\[
 \text{Loss}(x, y, \mathbf{w}) = \frac{1}{2} (\sigma(\mathbf{w} \cdot \phi(x)) - y)^2 
\]

**Part b: Gradient**
Let \(p = \sigma(\mathbf{w} \cdot \phi(x))\). The derivative of the sigmoid is \(\sigma'(z) = p(1 - p)\).
\[
 \nabla_\mathbf{w} \text{Loss} = (p - y) \cdot p(1 - p) \cdot \phi(x) 
\]

**Part c: Magnitude of Gradient**
To make the gradient arbitrarily small, we need \(p \to 1\) (since \(y=1\)). This happens when \(\mathbf{w} \cdot \phi(x) \to \infty\). Thus, making the weights arbitrarily large in the direction of \(\phi(x)\) will shrink the gradient. The gradient can **never be exactly zero** for finite weights because the sigmoid function only approaches 1 asymptotically. Since we cannot set weights to infinity, \(1-p\) will always be strictly greater than zero.

## Problem 3: Sentiment Classification

**Part e: Character N-grams**
For the value of \(n\), \(n=3\) or \(n=4\) usually produces the smallest validation error. Character n-grams perform almost as well as word features because they capture sub-word morphological patterns (like prefixes and suffixes) and handle misspellings or punctuation.
A review where character n-grams outperform word features: *"I hated this movie. It was un-freaking-believable."* Word features would see "un-freaking-believable" as one unknown word, but character n-grams (like "fre", "eak", "bel") can match it to the positive root word "believable", capturing the sentiment better.

## Problem 5: K-means Clustering

**Part a: Manual 2-means Execution**
*Run 1:* Initial \(\mu_1=[0,0], \mu_2=[11,0]\). Final centers: \(\mu_1=[2,0], \mu_2=[8.5,0]\). Assignments: \(x_1, x_2\) in cluster 1; \(x_3, x_4\) in cluster 2. Final loss: 20.5.
*Run 2:* Initial \(\mu_1=[0,0], \mu_2=[4,0]\). Final centers: \(\mu_1=[0,0], \mu_2=[7,0]\). Assignments: \(x_1\) in cluster 1; \(x_2, x_3, x_4\) in cluster 2. Final loss: 26.

**Part c: Scaling K-means**
*Scaling all dimensions by a non-zero factor:* **Yes**, the clusters remain the same. K-means uses Euclidean distance, and scaling all dimensions by \(c\) multiplies all distances by \(c^2\). Since the relative ordering of distances doesn't change, the nearest centroid remains identical.
*Scaling only certain dimensions:* **No**, the clusters are not guaranteed to remain the same. Scaling one dimension changes the relative distances between points. For example, points A=[1,1], B=[2,1], C=[1,10] might cluster differently if we multiply only the first dimension by 10.
