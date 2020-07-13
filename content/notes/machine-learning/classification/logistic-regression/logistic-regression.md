---
# Basic info
title: "Logistic Regression"
date: 2020-07-13
draft: false
# type: docs # page type
authors: ["admin"]
tags: ["ML", "Classification"]
categories: ["Machine Learning"]
toc: true # Show table of contents?

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: "Logistic Regression overview"
share: false  # Show social sharing links?
featured: true
lastmod: true

comments: false  # Show comments?
disable_comment: true
commentable: false  # Allow visitors to comment? Supported by the Page, Post, and Docs content types.

editable: false  # Allow visitors to edit the page? Supported by the Page, Post, and Docs content types.

# Optional header image (relative to `static/img/` folder).
header:
  caption: ""
  image: ""
---

💡 **Use regression algorithm for classification**

Logistic regression: **estimate the probability that an instance belongs to a particular class** 

- If the estimated probability is **greater than 50%**, then the model predicts that the instance belongs to that class (called the **positive** class, labeled “1”), 
- or else it predicts that it does not (i.e., it belongs to the **negative** class, labeled “0”). 

This makes it a **binary** classifier. 

## Logistic / Sigmoid function

<img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Sigmoid-function-2.svg" style="zoom:60%; background-color:white">

$\sigma(t)=\frac{1}{1+\exp (-t)}$
- Bounded: $\sigma(t) \in (0, 1)$

- Symmetric: $1 - \sigma(t) = \sigma(-t)$

- Derivative: $\sigma^{\prime}(t)=\sigma(t)(1-\sigma(t))$

  

## Estimating probabilities and making prediction

1. Computes a weighted sum of the input features (plus a bias term) 

2. Outputs the logistic of this result

    $\hat{p}=h_{\theta}(\mathbf{x})=\sigma\left(\mathbf{x}^{\mathrm{T}} \boldsymbol{\theta}\right)$

3. Prediction: 

    $\hat{y}=\left\{\begin{array}{ll}0 & \hat{p}<0.5\left(\Leftrightarrow h_{\theta}(\mathbf{x})<0\right) \\ 1 & \hat{p} \geq 0.5\left(\Leftrightarrow h_{\theta}(\mathbf{x}) \geq 0\right)\end{array}\right.$



## Train and cost function

Objective of training: to set the parameter vector $\boldsymbol{\theta}$ so that the model estimates:
- high probabilities ($\geq 0.5$) for positive instances ($y=1$)
- low probabilities ($< 0.5$) for negative instances ($y=0$)

### Cost function of a single training instance:

$c(\boldsymbol{\theta})=\left\{\begin{array}{cc}-\log (\hat{p}) & y=1 \\ -\log (1-\hat{p}) & y=0\end{array}\right.$

> <img src="https://miro.medium.com/max/1621/1*_NeTem-yeZ8Pr9cVUoi_HA.png" style="zoom:30%; background-color:white">
>
> - Actual lable: $y=1$, Misclassification: $\hat{y} = 0 \Leftrightarrow$ $\hat{p} = \sigma(h_{\boldsymbol{\theta}}(x))$ close to 0 $\Leftrightarrow c(\boldsymbol{\theta})$ large 
> - Actual lable: $y=0$, Misclassification: $\hat{y} = 1 \Leftrightarrow$ $\hat{p} = \sigma(h_{\boldsymbol{\theta}}(x))$ close to 1 $\Leftrightarrow c(\boldsymbol{\theta})$ large 

### The cost function over the whole training set

Simply the average cost over all training instances (Combining the expressions of two different cases above into one single expression):

$\begin{aligned} J(\boldsymbol{\theta}) &=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(\hat{p}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-\hat{p}^{(i)}\right)\right] \\\\ &=\frac{1}{m} \sum_{i=1}^{m}\left[-y^{(i)} \log \left(\hat{p}^{(i)}\right)-\left(1-y^{(i)}\right) \log \left(1-\hat{p}^{(i)}\right)\right] \end{aligned}$

> - $y^{(i)} =1:-y^{(i)} \log \left(\hat{p}^{(i)}\right)-\left(1-y^{(i)}\right) \log \left(1-\hat{p}^{(i)}\right)=-\log \left(\hat{p}^{(i)}\right)$
> - $y^{(i)} =0:-y^{(i)} \log \left(\hat{p}^{(i)}\right)-\left(1-y^{(i)}\right) \log \left(1-\hat{p}^{(i)}\right)=-\log \left(1-\hat{p}^{(i)}\right)$
> (Exactly the same as $c(\boldsymbol{\theta})$ for a single instance above 👏)

### Training 

- No closed-form equation 🤪

- But it is convex so Gradient Descent (or any other optimization algorithm) is guaranteed to find the global minimum     

- Partial derivatives of the cost function with regards to the $j$-th model parameter $\theta_j$:

    $$
    \frac{\partial}{\partial \theta_{j}} J(\boldsymbol{\theta})=\frac{1}{m} \displaystyle \sum_{i=1}^{m}\left(\sigma\left(\boldsymbol{\theta}^{T} \mathbf{x}^{(l)}\right)-y^{(i)}\right) x_{j}^{(i)}
    $$
    