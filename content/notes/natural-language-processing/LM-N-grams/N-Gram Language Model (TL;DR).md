---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 350

# Basic metadata
title: "Summary (TL;DR)"
date: 2020-08-03
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "Language models"]
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
        parent: LM-N-gram
        weight: 5

---

## **Language models** 

- offer a way to assign a probability to a sentence or other sequence of words, and to predict a word from preceding words.
  - $P(w|h)$: Probability of word $w$ given history $h$

## **n-gram model**

- estimate words from a fixed window of previous words
  $$
  P\left(w_{n} | w_{1}^{n-1}\right) \approx P\left(w_{n} | w_{n-N+1}^{n-1}\right)
  $$

- n-gram probabilities can be estimated by counting in a corpus and normalizing (**MLE**)
  $$
  P\left(w_{n} | w_{n-N+1}^{n-1}\right)=\frac{C\left(w_{n-N+1}^{n-1} w_{n}\right)}{C\left(w_{n-N+1}^{n-1}\right)}
  $$

- Evaluation

  - Extrinsically in some task (expensive!)

  - Instrinsically using **perplexity**

    - perplexity of a test set according to a language model: the geometric mean of the inverse test set probability computed by the model.
      $$
      \begin{array}{ll}
      \operatorname{PP}(W) &=P\left(w_{1} w_{2} \ldots w_{N}\right)^{-\frac{1}{N}} \\\\
      &=\sqrt[N]{\frac{1}{P\left(w_{1} w_{2} \ldots w_{N}\right)}} \\\\
      &\overset{\text{chain rule}}{=} \sqrt[N]{\displaystyle\prod_{i=1}^{N} \frac{1}{P\left(w_{i} | w_{1} \ldots w_{i-1}\right)}}
      \end{array}
      $$

## **Smoothing**

provide a more sophisticated way to estimate the probability of n-grams

- Laplace (Add-one) smmothing
  $$
  P_{\text {Laplace}}\left(w_{i}\right)=\frac{c_{i}+1}{N+V}
  $$

  - $V$: Number of words in the vocabulary

- Add-k smoothing
  $$
  P_{\mathrm{Add}-\mathrm{k}}^{*}\left(w_{n} | w_{n-1}\right)=\frac{C\left(w_{n-1} w_{n}\right)+k}{C\left(w_{n-1}\right)+k V}
  $$

- Backoff or interpolation

  - Rely on lower-order n-gram counts

- Kneser-Ney smoothing

  - makes use of the probability of a word being a novel continuation

