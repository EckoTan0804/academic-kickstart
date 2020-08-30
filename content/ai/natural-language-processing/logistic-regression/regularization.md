---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 550

# Basic metadata
title: "Regularization"
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
        weight: 5

---

## Overfitting

🔴 Problem with learning weights that make the model perfectly match the training data:

- If a feature is perfectly predictive of the outcome because it happens to only occur in one class, it will be assigned a very high weight. The weights for features will attempt to perfectly fit details of the training set, *in fact too perfectly*, modeling noisy factors that just accidentally correlate with the class. 🤪

This problem is called **overfitting**.

A good model should be able to **generalize well from the training data to the *unseen* test set**, but a model that overfits will have *poor* generalization. :cry:

## 🔧 Solution: Regularization

 Add a regularization term $R(\theta)$ to the objective function:
$$
\hat{\theta}=\underset{\theta}{\operatorname{argmax}} \sum_{i=1}^{m} \log P\left(y^{(i)} | x^{(i)}\right)-\alpha R(\theta)
$$

- $R(\theta)$: penalize large weights
  - a setting of the weights that matches the training data perfectly— but uses many weights with high values to do so—will be penalized more than a setting that matches the data a little less well, but does so using smaller weights.

Two common regularization terms:

- **L2 regularization** (Ridge regression)
  $$
  R(\theta)=\|\theta\|_{2}^{2}=\sum_{j=1}^{n} \theta_{j}^{2}
  $$

  - quadratic function of the weight values

  - $\|\theta\|_{2}^{2}$: L2 Norm, is the same as the Euclidean distance of the vector $\theta$ from the origin

  - L2 regularized objective function:
    $$
    \hat{\theta}=\underset{\theta}{\operatorname{argmax}}\left[\sum_{1=i}^{m} \log P\left(y^{(i)} | x^{(i)}\right)\right]-\alpha \sum_{j=1}^{n} \theta_{j}^{2}
    $$

- **L1 regularization** (Lasso regression)
  $$
  R(\theta)=\|\theta\|_{1}=\sum_{i=1}^{n}\left|\theta_{i}\right|
  $$

  - linear function of the weight values

  - $\|\theta\|_{1}$: L1 Norm, is the sum of the absolute values of the weights. 

    - Also called **Manhattan distance** (the Manhattan distance is the distance you’d have to walk between two points in a city with a street grid like New York)

  - L1 regularized objective function
    $$
    \hat{\theta}=\underset{\theta}{\operatorname{argmax}}\left[\sum_{1=i}^{m} \log P\left(y^{(i)} | x^{(i)}\right)\right]-\alpha \sum_{j=1}^{n}\left|\theta_{j}\right|
    $$

### L1- Vs. L2-Regularization

- L2 regularization is easier to optimize because of its simple derivative (the derivative of  $\theta^2$ is just $2\theta$), while L1 regularization is more complex ((the derivative of $|\theta|$ is non-continuous at zero)

  

- Where L2 prefers weight vectors with many small weights, L1 prefers sparse solutions with some larger weights but many more weights set to zero.

  - Thus L1 regularization leads to much sparser weight vectors (far fewer features).

Both L1 and L2 regularization have Bayesian interpretations as constraints on the prior of how weights should look.

- L1 regularization can be viewed as a **Laplace prior** on the weights.

- L2 regularization corresponds to assuming that weights are distributed according to a gaussian distribution with mean $μ = 0$.

  - In a gaussian or normal distribution, the further away a value is from the mean, the lower its probability (scaled by the variance $σ$)

  - By using a gaussian prior on the weights, we are saying that weights prefer to have the value 0. 

    A gaussian for a weight $\theta_j$ is:

    $\frac{1}{\sqrt{2 \pi \sigma_{j}^{2}}} \exp \left(-\frac{\left(\theta_{j}-\mu_{j}\right)^{2}}{2 \sigma_{j}^{2}}\right)$

    If we multiply each weight by a gaussian prior on the weight, we are thus maximizing the following constraint:
    $$
    \hat{\theta}=\underset{\theta}{\operatorname{argmax}} \prod_{i=1}^{M} P\left(y^{(i)} | x^{(i)}\right) \times \prod_{j=1}^{n} \frac{1}{\sqrt{2 \pi \sigma_{j}^{2}}} \exp \left(-\frac{\left(\theta_{j}-\mu_{j}\right)^{2}}{2 \sigma_{j}^{2}}\right)
    $$
    In log space, with $\mu=0$, and assuming $2\sigma^2=1$, we get:
    $$
    \hat{\theta}=\underset{\theta}{\operatorname{argmax}} \sum_{i=1}^{m} \log P\left(y^{(i)} | x^{(i)}\right)-\alpha \sum_{j=1}^{n} \theta_{j}^{2}
    $$

