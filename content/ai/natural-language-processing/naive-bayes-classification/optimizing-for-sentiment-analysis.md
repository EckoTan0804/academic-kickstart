---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 440

# Basic metadata
title: "Optimizing for Sentiment Analysis"
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
        weight: 4

---

While standard naive Bayes text classification can work well for sentiment analysis, some small changes are generally employed that improve performance. 💪

## Binary multinomial naive Bayes (binary NB)

First, for sentiment classification and a number of other text classification tasks, whether a word occurs or not seems to matter more than its frequency. 

**Thus it often improves performance to clip the word counts in each document at 1**.

Example:

- The example is worked without add-1 smoothing to make the differences clearer. 
- Note that the results counts need not be 1; the word *great* has a count of 2 even for Binary NB, because it appears in multiple documents (in both positive and negative class).

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-14%2012.55.28.png" alt="截屏2020-06-14 12.55.28" style="zoom:80%;" />

## Deal with negation

During text normalization, prepend the prefix *NOT_* to every word after a token of logical negation (*n’t, not, no, never*) until the next punc- tuation mark. 

Example:

![截屏2020-06-14 12.58.03](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-14%2012.58.03.png)

## Deal with insufficient labeled training data

Derive the positive and negative word features from sentiment lexicons, lists of words that are pre-annotated with positive or negative sentiment.

Popular lexicons:

- General Inquirer
- LIWC
- opinion lexicon
- MPQA Subjectivity Lexicon