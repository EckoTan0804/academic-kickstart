# Logistic Regression (TL;DR)

- Supervised classification

- Input: $x = (x_1, x_2, \dots, x_n)^T$

- Output: $y \in \{0, 1\}$

- Parameters:

  - Weight: $w = (w_1, w_2, \dots, w_n)^T$
  - Bias $b$

- Prediction
  $$
  z = w \cdot x + b \\
  P(y=1|x)=\sigma(z) = \frac{1}{1+e^{-z}}\\
  y=\left\{\begin{array}{ll}
  1 & \text { if } P(y=1 | x)>0.5 \\
  0 & \text { otherwise }
  \end{array}\right.
  $$

- Training/Learning

  - Loss function

    - For a single sample $x$
      $$
      \hat{y} = \sigma(w \cdot x + b)
      $$
      And we define $\hat{y}:=P(y=1|x)$

      $y \in \{0, 1\} \Rightarrow$
      $$
      P(y | x)=\left\{\begin{array}{lr}
      \hat{y} & y=1 \\
      1-\hat{y}  & y=0
      \end{array}\right.
      $$
      The probability of correct prediction can thus be expressed as:
      $$
      P(y|x)=\hat{y}^y (1-\hat{y})^{1-y}
      $$
      We want to maximize $P(y|x)$
      $$
      \begin{array}{ll}
      &\max \quad P(y|x) \\
      \equiv &\max \quad \log(P(y|x)) \\
      = &\max \quad \log(\hat{y}^y (1-\hat{y})^{1-y})\\
      = &\max \quad y\log(\hat{y}) + (1-y)\log(1-\hat{y}) \\
      \equiv &\min \quad -[y \log \hat{y}+(1-y) \log (1-\hat{y})] \\
      = &\min \quad \underbrace{-[y \log \sigma(w \cdot x + b) + (1-y) \log (1-\sigma(w \cdot x + b))]}_{=:L_{CE}(w, b)} \\
      \end{array}
      $$
      $L_{CE}(w, b)$ is called the **Cross-Entropy loss**.

    - For a mini-batch of samples of size $m$

      - $(x^{(i)}, y^{(i)})$: $i$-th Training sample

      - Loss function is the **average loss** for each example
        $$
        L(w, b) = =-\frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log \sigma\left(w \cdot x^{(i)}+b\right)+\left(1-y^{(i)}\right) \log \left(1-\sigma\left(w \cdot x^{(i)}+b\right)\right)
        $$

  - Algorithm: **Gradient descent**

    - Gradient for single sample:
      $$
      \frac{\partial L_{C E}(w, b)}{\partial w_{j}}=[\sigma(w \cdot x+b)-y] x_{j}
      $$

    - Gradient for mini-batch:
      $$
      \frac{\partial L(w, b)}{\partial w_{j}}=\frac{1}{m} \sum_{i=1}^{m}\left[\sigma\left(w \cdot x^{(i)}+b\right)-y^{(i)}\right] x_{j}^{(i)}
      $$

      - $x_j^{(i)}$: $j$-th feature of the $i$-th sample



## Multinomial Logistic Regression

- Also called **Softmax regression**, **MaxEnt classifier**

- Softmax function
  $$
  \operatorname{softmax}\left(z_{i}\right)=\frac{e^{z_{i}}}{\sum_{j=1}^{k} e^{z_{j}}} \qquad 1 \leq i \leq k
  $$

- Compute the probability of $y$ being in each potential class $c \in C$, $p(y=c|x)$, using softmax function:
  $$
  p(y=c | x)=\frac{e^{w_{c} \cdot x+b_{c}}}{\displaystyle\sum_{j=1}^{k} e^{w_{j} \cdot x+b_{j}}}
  $$

- Prediction
  $$
  \begin{array}{ll}
  \hat{c} &= \underset{c}{\arg \max} \quad p(y=c | x) \\
  &= \underset{c}{\arg \max} \quad \frac{e^{w_{c} \cdot x+b_{c}}}{\displaystyle\sum_{j=1}^{k} e^{w_{j} \cdot x+b_{j}}}
  \end{array}
  $$

- Learning

  - For a singnle sample $x$, the loss function is
    $$
    \begin{aligned}
    L_{C E}(w, b) &=-\sum_{k=1}^{K} 1\{y=k\} \log p(y=k | x) \\
    &=-\sum_{k=1}^{K} 1\{y=k\} \log \frac{e^{w_{k} \cdot x+b_{k}}}{\sum_{j=1}^{K} e^{w_{j} \cdot x+b_{j}}}
    \end{aligned}
    $$

    - $1\{\}$: evaluates to $1$ if the condition in the brackets is true and to $0$ otherwise.

  - Gradient
    $$
    \begin{aligned}
    \frac{\partial L_{C E}}{\partial w_{k}} &=-(1\{y=k\}-p(y=k | x)) x_{k} \\
    &=-\left(1\{y=k\}-\frac{e^{w_{k} \cdot x+b_{k}}}{\sum_{j=1}^{K} e^{w_{j} \cdot x+b_{j}}}\right) x_{k}
    \end{aligned}
    $$
    

