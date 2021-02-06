---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 610

# Basic metadata
title: "Why ensemble learning?"
date: 2020-11-07
draft: false
type: docs # page type
authors: ["admin"]
tags: ["ML", "Ensemble Learning"]
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
        parent: ensemble-learning
        weight: 1
---


**wisdom of the crowd** : In many cases you will find that this aggregated answer is better than an expert’s answer.

Similarly, if you aggregate the predictions of a group of predictors (such as classifiers or regressors), you will often get **better** predictions than with the best individual predictor. 

A group of predictors is called an **ensemble**; 

thus, this technique is called **Ensemble Learning**, 

and an Ensemble Learning algorithm is called an **Ensemble method**.

Popular Emsemble methods:
- [Bagging and Pasting]({{< relref "bagging-and-pasting.md" >}})
- [Boosting]({{< relref "boosting.md" >}})
- stacking
- [Voting Classifier]({{< relref "voting-classifier.md" >}})

