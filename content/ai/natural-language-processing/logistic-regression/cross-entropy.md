---
# Title, summary, and position in the list
linktitle: "Cross Entropy"
summary: ""
weight: 530

# Basic metadata
title: "The Cross-Entropy Loss Function"
date: 2020-08-03
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "Logistic Regression"]
categories: ["NLP"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: ""
share: false  # Show social sharing links?
featured: true

comments: false  # Show comments?
disable_comment: true
commentable: false  # Allow visitors to comment? Supported by the Page, Post, and Docs content types.

editable: false  # Allow visitors to edit the page? Supported by the Page, Post, and Docs content types.

# Optional header image (relative to `static/img/` folder).
header:
  caption: ""
  image: ""

# Menu
menu: 
    natural-language-processing:
        parent: logistic-reg
        weight: 3

---

## Motivation

We need a loss function that expresses, **for an observation $x$, how close the classifier output ($\hat{y}=\sigma(w \cdot x+b)$) is to the correct output ($y$, which is $0$ or $1$)**:
$$
L(\hat{y}, y)= \text{How much } \hat{y} \text{ differs from the true } y
$$
This loss function should prefer the correct class labels of the training examples to be *more likely*. 

👆 This is called **conditional maximum likelihood estimation**: we choose the parameters $w, b$ that maximize the log probability of the true $y$ labels in the training data given the observations $x$. The resulting loss function is the *negative* log likelihood loss, generally called the **cross-entropy loss**.

## Derivation

Task: for a single observation $x$,   learn weights that maximize $p(y|x)$, the probability of the correct label 

There're only two discretions outcomes ($1$ or $0$)

$\Rightarrow$ This is a **Bernoulli distribution**. The probability $p(y|x)$ for one observation can be expressed as:
$$
p(y | x)=\hat{y}^{y}(1-\hat{y})^{1-y}
$$

- $y=1, p(y|x)=\hat{y}$
- $y=0, p(y|x)=1-\hat{y}$

Now we take the log of both sides. This will turn out to be handy mathematically, and doesn’t hurt us (whatever values maximize a probability will also maximize the log of the probability):
$$
\begin{aligned}
\log p(y | x) &=\log \left[\hat{y}^{y}(1-\hat{y})^{1-y}\right] \\\\
&=y \log \hat{y}+(1-y) \log (1-\hat{y})
\end{aligned}
$$
👆 This is the log likelihood that should be maximized.

In order to turn this into loss function (something that we need to minimize), we’ll just flip the sign. The result is the **cross-entropy loss**: 
$$
L_{C E}(\hat{y}, y)=-\log p(y | x)=-[y \log \hat{y}+(1-y) \log (1-\hat{y})]
$$
Recall that $\hat{y}=\sigma(w \cdot x+b)$:
$$
L_{C E}(w, b)=-[y \log \sigma(w \cdot x+b)+(1-y) \log (1-\sigma(w \cdot x+b))]
$$

## Example

Let’s see if this loss function does the right thing for example above.

We want the loss to be 

- **smaller** if the model’s estimate is **close to correct**, and 

- **bigger** if the model is **confused**.

Let’s suppose the correct gold label for the sentiment example above is positive, i.e.: $y=1$.

- In this case our model is doing well 👏, since it gave the example a a higher probability of being positive ($0.70$) than negative ($0.30$).

  If we plug $\sigma(w \cdot x+b)=0.70$ and $y=1$ into the cross-entropy loss, we get

$$
\begin{aligned}
L_{C E}(w, b) &=-[y \log \sigma(w \cdot x+b)+(1-y) \log (1-\sigma(w \cdot x+b))] \\\\
&=-[\log \sigma(w \cdot x+b)] \\\\
&=-\log (0.70) \\\\
&=0.36
\end{aligned}
$$

By contrast, let's pretend instead that the example was negative, i.e.: $y=0$.

- In this case our model is confused 🤪, and we’d want the loss to be higher.

  If we plug $y=0$ and $1-\sigma(w \cdot x+b)=0.30$ into the cross-entropy loss, we get
  $$
  \begin{aligned}
  L_{C E}(w, b) &=-[y \log \sigma(w \cdot x+b)+(1-y) \log (1-\sigma(w \cdot x+b))] \\\\
  &= -[\log (1-\sigma(w \cdot x+b))] \\\\
  &=-\log (.31) \\\\
  &= 1.20
  \end{aligned}
  $$

It's obvious that the lost for the first classifier ($0.36$) is less than the loss for the second classifier ($1.17$).

## Why minimizing this negative log probability works?

A perfect classifier would assign probability $1$ to the correct outcome and probability $0$ to the incorrect outcome. That means: 

- the higher $\hat{y}$ (the closer it is to 1), the better the classifier; 
- the lower $\hat{y}$ is (the closer it is to 0), the worse the classifier. 

The negative log of this probability is a convenient loss metric since it goes from 0 (negative log of 1, no loss) to infinity (negative log of 0, infinite loss). This loss function also ensures that as the probability of the correct answer is maximized, the probability of the incorrect answer is minimized; since the two sum to one, any increase in the probability of the correct answer is coming at the expense of the incorrect answer.

