---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 500

# Basic metadata
title: "Generative and Discriminative Classifiers"
date: 2020-08-03
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "Logistic Regression"]
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
        parent: logistic-reg
        weight: 1

---

The most important difference between naive Bayes and logistic regression is that 

- logistic regression is a **discriminative** classifier while 
- naive Bayes is a **generative** classifier.

Consider a visual metaphor: imagine we’re trying to distinguish dog images from cat images.

-  Generative model
   - Try to understand what dogs look like and what cats look like
   - You might literally ask such a model to ‘generate’, i.e. draw, a dog
   - Given a test image, the system then asks whether it’s the cat model or the dog model that better fits (is less surprised by) the image, and chooses that as its label.
-  disciminative model
   - only trying to learn to distinguish the classes
   - So maybe all the dogs in the training data are wearing collars and the cats aren’t. If that one feature neatly separates the classes, the model is satisfied. If you ask such a model what it knows about cats all it can say is that they don’t wear collars. 

More formally, recall that the [naive Bayes]({{< relref"../naive-bayes-classification/naive-bayes-classifiers.md" >}}) assigns a class $c$ to a document $d$ NOT by directly computing $p(c|d)$, but by computing a likelihood and a prior.
$$
\hat{c}=\underset{c \in C}{\operatorname{argmax}} \overbrace{P(d | c)}^{\text { likelihood prior }} \overbrace{P(c)}^{\text { prior }}
$$

- Generative model (like naive Bayes)
  - Makes use of the likelihood term
    - Expresses how to generate the features of a document *if we knew it was of class* $c$

- Discriminative model
  -  attempts to directly compute $P(c|d)$
  -  It will learn to assign a high weight to document features that directly improve its ability to *discriminate* between possible classes, even if it couldn’t generate an example of one of the classes.

### Components of a probabilistic machine learning classifier

- Training corpus of $M$ input/output pairs $(x^{(i)}, y^{(i)})$

- A **feature representation** of the input

  - For each input observation $x^{(i)}$, this will be a vector of features $[x_1, x_2, \dots, x_n]$
    - $x_{i}^{(j)}$: feature $i$ for input $x^{(j)}$

- A **classification function** that computes $\hat{y}$, the estimated class, via $p(y|x)$

- An objective function for learning, usually involving minimizing error on

  training examples

- An algorithm for optimizing the objective function.

Logistic regression has two phases:

- **training**: we train the system (specifically the weights $w$ and $b$) using stochastic gradient descent and the cross-entropy loss.

- **test**: Given a test example $x$ we compute $p(y|x)$ and return the higher probability label $y=1$ or $y=0$.