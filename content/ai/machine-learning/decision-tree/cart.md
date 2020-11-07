---
# Title, summary, and position in the list
linktitle: "CART"
summary: ""
# weight: 

# Basic metadata
title: "Classification And Regression Tree (CART)"
date: 2020-10-27
draft: false
type: docs # page type
authors: ["admin"]
tags: ["ML", "Classification", "Regression", "Decision Tree"]
categories: ["Machine Learning"]
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
    machine-learning:
        parent: decision-tree
        weight: 1
---

## Tree-based Methods

**CART**: **C**lassification **A**nd **R**egression **T**ree

### Grow a binary tree

- At each node, “split” the data into two “daughter” nodes.
- Splits are chosen using a splitting criterion.
- Bottom nodes are “terminal” nodes.

|                    | Type of tree    | Predicted value at a node                                    | Split criterion                                              |
| ------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Regression**     | Regression tree | The predicted value at a node is the **average response** variable for all observations in the node | **Minimum residual sum of squares** <br />$$\mathrm{RSS}=\sum_{\text {left }}\left(y_{i}-\bar{y}_{L}\right)^{2}+\sum_{\text {right }}\left(y_{i}-\bar{y}_{R}\right)^{2}$$<li />$\bar{y}_L$ / $\bar{y}_R$: average label values in the left / right subtree <br />(Split such that variance in subtress is minimized) |
| **Classification** | Decision tree   | The predicted class is the **most common class** in the node (majority vote). | **Minimum entropy** in subtrees<br />$$\text { score }=N_{L} H\left(p_{\mathrm{L}}\right)+N_{R} H\left(p_{\mathrm{R}}\right)$$<li />$H\left(p_{L}\right)=-\sum_{k} p_{L}(k) \log p_{L}(k)$: entropy in the left sub-tree <li /> $p_L(k)$: proportion of class $k$ in left tree<br />(Split such that class-labels in sub-trees are "pure") |

### When stop?

**Stop if:**

- Minimum number of samples per node
- Maximum depth 

... has been reached

(Both criterias again influence the **complexity** of the tree)

### Controlling the tree complexity

| Number of samples per leaf | Affect                              |                                                              |
| -------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| **Small**                  | Tree is **very sensitive** to noise | <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/屏幕快照%202020-03-01%2023.26.23.png" alt="屏幕快照 2020-03-01 23.26.23" style="zoom:33%;" /><br /><img src="https://github.com/EckoTan0804/upic-repo/blob/master/uPic/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202020-03-01%2023.25.40.png?raw=true" alt="屏幕快照 2020-03-01 23.25.40.png" style="zoom:33%;" /> |
| **Large**                  | Tree is **not expressive enough**   | <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/屏幕快照%202020-03-01%2023.25.50.png" alt="屏幕快照 2020-03-01 23.25.50" style="zoom:33%;" /> |



### Advantages 👍

- Applicable to both regression and classification problems.

- Handle categorical predictors naturally.

- Computationally simple and quick to fit, even for large problems.

- No formal distributional assumptions (non-parametric).

- Can handle highly non-linear interactions and classification boundaries.

- Automatic variable selection.

- Very easy to interpret if the tree is small.

### Disadvantages 👎

- ***Accuracy*** 

  current methods, such as support vector machines and ensemble classifiers often have 30% lower error rates than CART.

- ***Instability*** 

  if we change the data a little, the tree picture can change a lot. So the interpretation is not as straightforward as it appears.