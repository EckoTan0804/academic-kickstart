---
# Basic info
title: "Logistic Regression in Probabilistic view"
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
summary: "Derivation of logistic regression in the view Probabilistic" 
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

# Menu
# menu: 
#     machine-learning:
#         parent: classification
#         weight: 2
---


Class label: 

$$
y_i \in \\{0, 1\\}
$$


Conditional probability distribution of the class label is

$$
\begin{aligned}
p(y=1|\boldsymbol{x}) &= \sigma(\boldsymbol{w}^T\boldsymbol{x}+b) \\\\
p(y=0|\boldsymbol{x}) &= 1 - \sigma(\boldsymbol{w}^T\boldsymbol{x}+b)
\end{aligned}
$$
with 

$$
\sigma(x) = \frac{1}{1+\operatorname{exp}(-x)}
$$


This is a **conditional Bernoulli distribution**. Therefore, the probability can be represented as

$$
\begin{array}{ll}
p(y|\boldsymbol{x}) &= p(y=1|\boldsymbol{x})^y p(y=0|\boldsymbol{x})^{1-y} \\\\
& = \sigma(\boldsymbol{w}^T\boldsymbol{x}+b)^y (1 - \sigma(\boldsymbol{w}^T\boldsymbol{x}+b))^{1-y}
\end{array}
$$


The **conditional Bernoulli log-likelihood** is (assuming training data is i.i.d)

$$
\begin{aligned}
\operatorname{loglik}(\boldsymbol{w}, \mathcal{D}) 
&= \log(\operatorname{lik}(\boldsymbol{w}, \mathcal{D})) \\\\
&= \log(\displaystyle\prod_i p(y_i|\boldsymbol{x}_i)) \\\\
&= \log\left(\displaystyle\prod_i \sigma(\boldsymbol{w}^T\boldsymbol{x}_i+b)^y \left(1 - \sigma(\boldsymbol{w}^T\boldsymbol{x}_i+b)\right)^{1-y}\right) \\\\
&= \displaystyle\sum_i y\log\left(\sigma(\boldsymbol{w}^T\boldsymbol{x}_i+b)\right)+ (1-y)\log\left(1 - \sigma(\boldsymbol{w}^T\boldsymbol{x}_i+b)\right) 
\end{aligned}
$$


Let 

$$
\tilde{\boldsymbol{w}}=\left(\begin{array}{c}1 \\\\ \boldsymbol{w} \end{array}\right), \quad \tilde{\boldsymbol{x}_i}=\left(\begin{array}{c}b \\\\ \boldsymbol{x}_i \end{array}\right)
$$


Then:

$$
\operatorname{loglik}(\boldsymbol{w}, \mathcal{D}) = \operatorname{loglik}(\tilde{\boldsymbol{w}}, \mathcal{D})  = \displaystyle\sum_i y\log\left(\sigma(\tilde{\boldsymbol{w}}^T\tilde{\boldsymbol{x}_i})\right)+ (1-y)\log\left(1 - \sigma(\tilde{\boldsymbol{w}}^T\tilde{\boldsymbol{x}_i}))\right)
$$


Our objective is to find the $\tilde{\boldsymbol{w}}^*$ that **maximize the log-likelihood**, i.e.

$$
\begin{array}{cl}
\tilde{\boldsymbol{w}}^* &= \underset{\tilde{\boldsymbol{w}}}{\arg \max} \quad \operatorname{loglik}(\tilde{\boldsymbol{w}}, \mathcal{D}) \\\\
&= \underset{\tilde{\boldsymbol{w}}}{\arg \min} \quad -\operatorname{loglik}(\tilde{\boldsymbol{w}}, \mathcal{D})\\\\
&= \underset{\tilde{\boldsymbol{w}}}{\arg \min} \quad \underbrace{-\left(\displaystyle\sum_i y\log\left(\sigma(\tilde{\boldsymbol{w}}^T\tilde{\boldsymbol{x}_i})\right) + (1-y)\log\left(1 - \sigma(\tilde{\boldsymbol{w}}^T\tilde{\boldsymbol{x}_i}))\right)\right)}_{\text{cross-entropy loss}}
\end{array}
$$


In other words, **maximizing the (log-)likelihood is the same as minimizing the cross entropy.**