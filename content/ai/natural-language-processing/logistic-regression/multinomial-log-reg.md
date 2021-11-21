---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 560

# Basic metadata
title: "Multinomial Logistic Regression"
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
        weight: 6

---

## Motivation

More than two classes? 

Use **multinomial logistic regression** (also called **softmax regression**, or **maxent classifier**). The target $y$ is a variable that ranges over **more than two** classes; we want to know the probability of $y$ being in each potential class $c \in C, p(y=c|x)$.

We use the **softmax** function to compute $p(y=c|x)$:

- Takes a vector $z=[z_1, z_2,\dots, z_k]$ of $k$ arbitrary values
- Maps them to a probability distribution
  - Each value $\in (0, 1)$
  - All the values summing to $1$

For a vector $z$ of dimensionality $k$, the softmax is:
$$
\operatorname{softmax}\left(z_{i}\right)=\frac{e^{z_{i}}}{\sum_{j=1}^{k} e^{z_{j}}} \qquad 1 \leq i \leq k
$$
The softmax of an input vector $z=[z_1, z_2,\dots, z_k]$ is thus:
$$
\operatorname{softmax}(z)=\left[\frac{e^{z_{1}}}{\sum_{i=1}^{k} e^{z_{i}}}, \frac{e^{z_{2}}}{\sum_{i=1}^{k} e^{z_{i}}}, \ldots, \frac{e^{z_{k}}}{\sum_{i=1}^{k} e^{z_{i}}}\right]
$$

- The denominator $\sum_{j=1}^{k} e^{z_{j}}$ is used to normalize all the values into probabilities.

Like the sigmoid, the input to the softmax will be the dot product between a weight vector $w$ and an input vector $x$ (plus a bias). But now we’ll need separate weight vectors (and bias) for each of the $K$ classes.
$$
p(y=c | x)=\frac{e^{w_{c} \cdot x+b_{c}}}{\displaystyle\sum_{j=1}^{k} e^{w_{j} \cdot x+b_{j}}}
$$

## Features in Multinomial Logistic Regression

For multiclass classification, input features are:

- observation $x$
- candidate output class $c$

$\Rightarrow$ When we are discussing features we will use the notation $f_i(c, x)$: feature $i$ for a particular class $c$ for a given observation $x$

### **Example**

Suppose we are doing text classification, and instead of binary classification our task is to assign one of the 3 classes +, −, or 0 (neutral) to a document. Now a feature related to exclamation marks might have a negative weight for 0 documents, and a positive weight for + or − documents:

![截屏2020-05-29 15.59.37](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-29%2015.59.37-20200803151242332.png)

## Learning in Multinomial Logistic Regression

The loss function for a single example $x$ is the sum of the logs of the $K$ output classes:
$$
\begin{aligned}
L_{C E}(\hat{y}, y) &=-\sum_{k=1}^{K} \mathbb{1}\\{y=k\\} \log p(y=k | x) \\\\
&=-\sum_{k=1}^{K} \mathbb{1}\\{y=k\\} \log \frac{e^{w_{k} \cdot x+b_{k}}}{\sum_{j=1}^{K} e^{w_{j} \cdot x+b_{j}}}
\end{aligned}
$$

- $1\{\}$: evaluates to $1$ if the condition in the brackets is true and to $0$ otherwise.

Gradient:
$$
\begin{aligned}
\frac{\partial L_{C E}}{\partial w_{k}} &=-(\mathbb{1}\\{y=k\\}-p(y=k | x)) x_{k} \\\\
&=-\left(\mathbb{1}\\{y=k\\}-\frac{e^{w_{k} \cdot x+b_{k}}}{\sum_{j=1}^{K} e^{w_{j} \cdot x+b_{j}}}\right) x_{k}
\end{aligned}
$$

