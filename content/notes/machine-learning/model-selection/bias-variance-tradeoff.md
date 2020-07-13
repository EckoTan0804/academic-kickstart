---
# Basic info
title: "Bias Variance Tradeoff"
date: 2020-07-06
draft: false
type: docs # page type
authors: ["admin"]
tags: ["ML", "Model Selection"]
categories: ["Machine Learning"]
toc: true # Show table of contents?

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
# summary: "Objective function overview"
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
menu: 
    machine-learning:
        parent: model-selection
        weight: 2
---

## TL;DR

|                    | Resaon                                     | Example                                | affect       | Model's complexity ⬆️ | Model's complexity ⬇️ |
| :----------------- | :----------------------------------------- | -------------------------------------- | ------------ | :------------------- | :------------------- |
| Bias               | wrong  assumption                          | assume a  quadratic model to be linear | underfitting | ⬇️                    | ⬆️                    |
| Variance           | excessive  sensitivity to small variations | high-degree  polynomial model          | overfitting  | ⬆️                    | ⬇️                    |
| Inreducible  error | noisy  data                                |                                        |              |                      |                      |

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/image-20200120105846503.png" alt="image-20200120105846503" style="zoom:50%;" />

## Explaination

A model’s generalization error can be expressed as the sum of three very different errors:

### Bias
This part of the generalization error is due to **wrong assumptions**, such as assuming that the data is linear when it is actually quadratic. 
A high-bias model is most likely to **underfit** the training data.

### Variance
This part is due to the model’s **excessive sensitivity to small variations** in the training data. \
A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have **high variance**, and thus to **overfit** the training data.

### Irreducible Error
This part is due to the **noisiness of the data** itself. 
The only way to reduce this part of the error is to **clean up the data** (e.g., fix the data sources, such as broken sensors, or detect and remove outliers). 

|               | High bias                      | Low bias               |
| ------------- | ------------------------------ | ---------------------- |
| High variance | something is terribly wrong! 😭 | Overfitting            |
| Low variance  | Underfitting                   | too good to be true! 🤪 |