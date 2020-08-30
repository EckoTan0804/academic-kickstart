---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 140

# Basic metadata
title: "Generalization"
date: 2020-08-17
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Optimization", "Nerual Network Basics"]
categories: ["Deep Learning"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?

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
    deep-learning:
        parent: nn-basics
        weight: 4

---

**Generalization: Ability to Apply what was learned during Training to *new (Test)* Data**

## Reasons for bad generalization

- Overfitting/Overtraining (trained too long)
- Too little training material
- Too many Parameters (weights) or inappropriate network architecture error

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2012.59.51.png" alt="截屏2020-08-17 12.59.51" style="zoom:67%;" />

## Prevent Overfitting

- The obviously best approach: Collect More Data! :muscle:
- If Data is Limited
  - Simplest Method for Best Generalization: **Early Stopping**
  - Optimize Parameters/Arcitecture
    - Architectural Learning
    - Choose Best Architecture by Repeated Experimentation on Cross Validation Set
    - Reduce Architecture Starting from Large
    - [Grow Architecture Starting from Small](#constructive-methods)

### Destructive Methods

**Reduce** Complexity of Network through **Regularization** 

- Weight Decay

- Weight Elimination
- [Optimal Brain Damage](#optimal-brain-damage)
- Optimal Brain Surgeon

#### Optimal Brain Damage

- 💡Idea: Certain connections are removed from the network to reduce complexity and to avoide overfitting
- Remove those connections that have the **least** effect on the Error (MSE, ..), i.e. are the least important.
  - But this is time consuming (difficult) 🤪

### Constructive Methods

Iteratively **Increasing/Growing** a Network (construktive) starting from a very small one

- [Cascade Correlation](#cascade-correlation)
- Meiosis Netzwerke
- ASO (Automativ Structure Optimization)

#### Cascade Correlation 

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2013.22.42.png" alt="截屏2020-08-17 13.22.42" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2013.23.01.png" alt="截屏2020-08-17 13.23.01" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2013.23.22.png" alt="截屏2020-08-17 13.23.22" style="zoom:50%;" />

- Adding a hidden unit
  - Input connections from all input units and from all already existing hidden units
  - First only these connections are adapted
  - Maximize the correlation between the activation of the candidate units and the residual error of the net
- Not necessary to determine the number of hidden units empirically
- Can produce deep networks without dramatic slowdown (bottom up, constructive learning)
- At each point only one layer of connections is trained 
- Learning is fast

- Learning is incremental

### Dropout

- Popular and very effective method for generalization

- 💡Idea

  - *Randomly* drop out (zero) hidden units and input features during training
  - Prevents feature co-adaptation

- Illustration

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2013.27.32.png" alt="截屏2020-08-17 13.27.32" style="zoom:67%;" />

- Dropout training & test

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2013.28.32.png" alt="截屏2020-08-17 13.28.32" style="zoom:67%;" />

