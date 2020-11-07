---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
# weight: 

# Basic metadata
title: "Boosting"
date: 2020-11-07
draft: false
type: docs # page type
authors: ["admin"]
tags: ["ML", "Classification", "Ensemble Learning"]
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
        weight: 5
---


# Boosting

Refers to any Ensemble method that can **combine serval weak learners into a strong learner**

💡 **General idea: train predictors sequentially, each trying to correct its predecessor.**

Popular boosting methods:
- [AdaBoost]({{< relref "adaboost.md" >}})
- Gradient Boost

