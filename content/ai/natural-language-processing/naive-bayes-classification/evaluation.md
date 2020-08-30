---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 450

# Basic metadata
title: "Evaluation"
date: 2020-08-03
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "Naive Bayes"]
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
        parent: navie-bayes
        weight: 5

---

## Two classes

### **Gold labels**

the human-defined labels for each document that we are trying to match

### Confusion Matrix

To evaluate any system for detecting things, we start by building a **Contingency table (Confusion matrix)**:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-14%2013.42.20.png" alt="截屏2020-06-14 13.42.20" style="zoom:80%;" />

### Evaluation metric

- **Accuracy**
  $$
  \text{Accuracy} = \frac{\text{tp+tn}}{\text{tp+fp+tn+fn}}
  $$

  - doesn’t work well when the classes are unbalanced 🤪

- **Precision (P)**

  - measures the percentage of the items that the system detected (i.e., the system labeled as positive) that are in fact positive (i.e., are positive according to the human gold labels).
    $$
    \text{Precision} = \frac{\text{tp}}{\text{tp+fp}}
    $$

- **Recall (R)**

  - measures the percentage of items actually present in the input (i.e., are positive according to the human gold labels) that were correctly identified by the system (i.e., the system labeled as positive).
    $$
    \text{Recall} = \frac{\text{tp}}{\text{tp+fn}}
    $$

- **F-measure**
  $$
  F_{\beta}=\frac{\left(\beta^{2}+1\right) P R}{\beta^{2} P+R}
  $$

  - $\beta$: differentially weights the importance of recall and precision, based perhaps on the needs of an application

    - $\beta > 1$: favor recall

    - $\beta < 1$: favor precision

    - $\beta = 1$: precision and recall are equally balanced (the most frequently used metric)
      $$
      F_{1}=\frac{2 P R}{P+R}
      $$

### More than two classes

Solution: **one-of** or **multinomial classification**

- The classes are **mutually exclusive** and each document or item appears in exactly ONE class.
- How it works?
  - We build a separate binary classifier trained on positive examples from $c$ and negative examples from all other classes. 
  - Given a test document or item $d$, we run all the classifiers and choose the label from the classifier with the highest score.

E.g.:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-14%2013.58.26.png" alt="截屏2020-06-14 13.58.26" style="zoom: 120%;" />

Evaluation metric:

- **Macroaveraging**
  1. compute the performance for each class
  2. then average over classes
- **Microaveraging**
  1. collect the decisions for all classes into a single contingency table
  2. then compute precision and recall from that table.

E.g.: 

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-14 14.00.23.png" alt="截屏2020-06-14 14.00.23" style="zoom:150%;" />

