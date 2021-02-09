---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 720

# Basic metadata
title: "LDA Summary"
date: 2020-11-07
draft: false
type: docs # page type
authors: ["admin"]
tags: ["ML", "Classification", "Non-parametric"]
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
        parent: non-parametric
        weight: 2
---

**Linear Discriminant Analysis (LDA)**

- also called **Fisher’s Linear Discriminant**

- reduces dimension (like PCA)
- but focuses on **maximizing seperability among known categories**



## 💡 Idea

1. Create a new axis
2. Project the data onto this new axis in a way to maximize the separation of two categories



## How it works?

### Create a new axis

According to two criteria (considered simultaneously):

- Maximize the distance between means

- Minimize the variation $s^2$ (which LDA calls "scatter") within each category

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-14%2015.11.22.png" alt="截屏2020-05-14 15.11.22" style="zoom:50%;" />

We have:
$$
\frac{(\overbrace{\mu_1 - \mu_2}^{=: d})^2}{s_1^2 + s_2^2} \qquad\left(\frac{\text{''ideally large''}}{\text{"ideally small"}}\right)
$$
**Why both distance and scatter are important?**

![截屏2020-05-14 15.17.59](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-14%2015.17.59.png)

#### More than 2 dimensions

The process is the **same** 👏:

Create an axis that maximizes the distance between the means for the two categories while minimizing the scatter

#### More than 2 categories (e.g. 3 categories)

Little difference:

- Measure the distances among the means

  - Find the point that is **central** to all of the data

  - Then measure the distances between a point that is central in each category and the main central point

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-14%2015.26.35.png" alt="截屏2020-05-14 15.26.35" style="zoom:50%;" />

  - Maximize the distance between each category and the central point while minimizing the scatter for each category

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-14%2015.28.40.png" alt="截屏2020-05-14 15.28.40" style="zoom:50%;" />

- Create 2 axes to separate the data (because the 3 central points for each category define a plane)

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-14%2015.30.16.png" alt="截屏2020-05-14 15.30.16" style="zoom:50%;" />



## LDA and PCA

### Similarities

- **Both rank the new axes in order of importance**
  - PC1 (the first new axis that PCA creates) accounts for the most variation in the data
    - PC2 (the second new axis) does the second best job
  - LD1 (the first new axis that LDA creates) accounts for the most variation between the categories
    - LD2 does the second best job
- **Both can let you dig in and see which features are driving the new axes**

- **Both try to reduce dimensions**
  - PCA looks at the features with the most variation
  - LDA tries to maximize the separation of known categories



## Reference

- https://www.youtube.com/watch?v=azXCzI57Yfc