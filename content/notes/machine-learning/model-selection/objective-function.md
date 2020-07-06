---
title: "Objective Function"
date: 2020-07-06
draft: false
authors: ["admin"]
tags: ["ML", "model selection"]
categories: ["Machine Learning"]
toc: true
disable_comment: true
reading_time: true
summary: "Objective function overview"
featured: true
lastmod: true
---


Objective function:

$$
\operatorname{Obj}(\Theta)= \overbrace{L(\Theta)}^{\text {Training Loss}}  + \underbrace{\Omega(\Theta)}_{\text{Regularization}}
$$

- Training loss: measures how well the model fit on training data
  $$
  L=\sum_{i=1}^{n} l\left(y_{i}, g_{i}\right)
  $$

  - Square loss: 
    $$
    l(y_i, \hat{y}_i) = (y_i - \hat{y}_i)^2
    $$
  - Logistic loss: 
    $$
    l(y_i, \hat{y}_i) = y_i \log(1 + e^{-\hat{y}_i}) + (1 - y_i) \log(1 + e^{\hat{y}_i})
    $$

- Regularization: How complicated is the model?
    - $L_2$ norm (Ridge): $\omega(w) = \lambda \|w\|^2$
    - $L_1$ norm (Lasso): $\omega(w) = \lambda \|w\|$
    

|                         | Objective Function                                           | Linear model? | Loss     | Regularization |
| ----------------------- | ------------------------------------------------------------ | ------------- | -------- | -------------- |
| **Ridge** regression    | $\sum_{i=1}^{n}\left(y_{i}-w^{\top} x_{i}\right)^{2}+\lambda\|w\|^{2}$ | ✅             | square   | $L_2$          |
| **Lasso** regression    | $\sum_{i=1}^{n}\left(y_{i}-w^{\top} x_{i}\right)^{2}+\lambda\|w\|$ | ✅             | square   | $L_2$          |
| **Logistic** regression | $\sum_{i=1}^{n}\left[y_{i} \cdot \ln \left(1+e^{-w^{\top} x_{i}}\right)+\left(1-y_{i}\right) \cdot \ln \left(1+e^{w^{\top} x_{i}}\right)\right]+\lambda\|w\|^{2}$ | ✅             | logistic | $L_1$          |



## Why do we want to contain two component in the objective? 

- **Optimizing training loss encourages predictive models** 

    - *Fitting well in training data at least get you close to training data which is hopefully close to the underlying distribution* 

- **Optimizing regularization encourages simple models** 

    - *Simpler models tends to have smaller variance in future predictions, making prediction* *stable* 