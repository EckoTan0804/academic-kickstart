---
# Basic info
title: "Cross Validation"
date: 2020-07-06
draft: false
type: docs
authors: ["admin"]
tags: ["ML", "Model Selection"]
categories: ["Machine Learning"]
toc: true

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: "Objective function overview"
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

menu: 
    machine-learning:
        parent: model-selection
        weight: 3
---


<img src="https://scikit-learn.org/stable/_images/grid_search_cross_validation.png" style="zoom:60%; background-color:white">

|                         | How it works?                                                | Illustration                                                 |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **K-fold**              | 1. Create $k$-fold partition of the dataset<br />2. Estimate $k$ hold-out predictors using $1$ partition as validation and $k-1$ partition as training set | <br /><img src="https://miro.medium.com/max/5535/1*QDH0DSCecArPmzQtEBh0yg.png" alt="img" style="zoom: 20%; background-color:white" /> |
| **Leave-One-Out (LOO)** | **(Special case with $k=n$)** <br />Consequently estimate $n$ hold-out predictors using $1$ partition as validation and $n-1$ partition as training set | <br /><img src="https://miro.medium.com/max/5284/1*9bs3OMsKOJntR8blRnVE9g.png" alt="img" style="zoom:20%; background-color:white" /><br /> |
| **Random sub-sampling** | 1. Randomly sample a fraction of $\alpha \cdot n, \alpha \in (0,1)$ data points for validation<br />2. Train on remaining points and validate, repeat $K$ times |                                                              |



## 🎥 Explaination

{{< youtube fSytzGwwBVw >}}

