---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 310

# Basic metadata
title: "Evaluating Language Models"
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
        weight: 2

---



## **Extrinsic evaluation**

- Best way to evaluate the performance of a language model

- Embed LM in an application and measure how much the application improves
- For speech recognition, we can compare the performance of two language models by running the speech recognizer twice, once with each language model, and seeing which gives the more accurate transcription.
- 🔴 <span style="color:red">Problem: running big NLP systems end-to-end is often very expensive</span>

## **Intrinsic evaluation**

- measures the quality of a model independent of any application.

- Can be used to quickly evaluate potential improvements in a language model
- We need
  - **Training set (Training corpus)**
  - **Development test set (devset)**
    - Also called **Validation set** (see [wiki](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets))
    - Particular test set
      - Implicitly tune to its characteristics
  - **Test set (Test corpus)**
    - NOT to let the test sentences into the training set!
    - Truely UNSEEN!
  - In practice: divide data into 80% training, 10% development, and 10% test.
- How it works?
  - Given a corpus of text and we want to compare two different n-gram models
    1. we divide the data into training and test sets, 
    2. train the parameters of both models on the training set, and 
    3. then compare how well the two trained models fit the test set.
       - "Fit the test set" means: whichever model assigns a **higher probability** to the test set—meaning it more accurately predicts the test set—is a **better** model.

## Perplexity

Instead of raw probability as our metric for evaluating language models, in practice we use **perplexity**.

The **perplexity** (sometimes called ***PP*** for short) of a language model on a test set is the inverse probability of the test set, normalized by the number of words.

For a test set $W=w_{1} w_{2} \ldots w_{N}$:
$$
\begin{array}{ll}
\operatorname{PP}(W) &=P\left(w_{1} w_{2} \ldots w_{N}\right)^{-\frac{1}{N}} \\\\
&=\sqrt[N]{\frac{1}{P\left(w_{1} w_{2} \ldots w_{N}\right)}} \\\\
&\overset{\text{chain rule}}{=} \sqrt[N]{\displaystyle\prod_{i=1}^{N} \frac{1}{P\left(w_{i} | w_{1} \ldots w_{i-1}\right)}}
\end{array}
$$
Thus, perplexity of *W* with a bigram language model is
$$
\operatorname{PP}(W)=\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P\left(w_{i} | w_{i-1}\right)}}
$$
The higher the conditional probabil- ity of the word sequence, the lower the perplexity. Thus, minimizing perplexity is equivalent to maximizing the test set probability according to the language model.

What we generally use for word sequence in perplexity computation is the ENTIRE sequence of words in test test. Since this sequence will cross many sentence boundaries, we need to include 

- the begin- and end-sentence markers `<s>` and `</s>` in the probability computation. 
- the end-of-sentence marker `</s>` (but not the beginning-of-sentence marker `<s>`) in the total count of word tokens *N*.

### Another aspect 

We can also think about perpleixty as the **weighted average branching factor** of a language.

- branching factor of a language: the number of possible next words that can follow any word.

Example

Consider the task of recognizing the digits in English (zero, one, two,..., nine), given that (both in some training set and in some test set) each of the 10 digits occurs with equal probability $P=\frac{1}{10}$. The perplexity of this mini-language is in fact 10. 
$$
\begin{aligned}
\operatorname{PP}(W) &=P\left(w_{1} w_{2} \ldots w_{N}\right)^{-\frac{1}{N}} \\\\
&=\left(\frac{1}{10}^{N}\right)^{-\frac{1}{N}} \\\\
&=\frac{1}{10}^{-1} \\\\
&=10
\end{aligned}
$$
Now suppose that the number zero is really frequent and occurs far more often than other numbers.

- 0 occur 91 times in the training set, and 
- each of the other digits occurred 1 time each.

Now we see the following test set: `0 0 0 0 0 3 0 0 0 0`. We should expect the perplexity of this test set to be lower since most of the time the next number will be zero, which is very predictable (i.e. has a high probability).  Thus, although the branching factor is still 10, the perplexity or *weighted* branching factor is smaller. 