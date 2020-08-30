---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 350

# Basic metadata
title: "Perplexity’s Relation to Entropy"
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

{{% alert note %}} 

**Recall**

A better n-gram model is one that assigns a higher probability to the test data, and perplexity is a normalized version of the probability of the test set.

{{% /alert %}}

**Entropy**: a measure of information

- Given:

  - A random variable $X$ ranging over whatever we are predicting (words, letters, parts of speech, the set of which we’ll call $χ$)
  - with a particular probability function $p(x)$

- The entropy of the random variable $X$ is
  $$
  H(X)=-\sum\_{x \in \chi} p(x) \log\_{2} p(x)
  $$

  - If we use log base 2, the resulting value of entropy will be measured in **bits**.

💡 Intuitive way to think about entropy: a **lower bound** on the number of bits it would take to encode a certain decision or piece of information in the optimal coding scheme.

**Example**

Imagine that we want to place a bet on a horse race but it is too far to go all the way to Yonkers Racetrack, so we’d like to send a short message to the bookie to tell him which of the eight horses to bet on.

One way to encode this message is just to use the binary representation of the horse’s number as the code: horse 1 would be `001`, horse 2 `010`, horse 3 `011`, and so on, with horse 8 coded as `000`. On average we would be sending 3 bits per race.

Suppose that the spread is the actual distribution of the bets placed and that we represent it as the prior probability of each horse as follows:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-04%2010.23.19.png" alt="截屏2020-06-04 10.23.19" style="zoom:80%;" />

The entropy of the random variable *X* that ranges over horses gives us a lower bound on the number of bits and is
$$
\begin{aligned}
H(X) &=-\sum\_{i=1}^{i=8} p(i) \log p(i) \\\\
&=-\frac{1}{2} \log \frac{1}{2}-\frac{1}{4} \log \frac{1}{4}-\frac{1}{8} \log \frac{1}{8}-\frac{1}{16} \log \frac{1}{16}-4\left(\frac{1}{64} \log \frac{1}{64}\right) \\\\
&=2 \text { bits }
\end{aligned}
$$
A code that averages 2 bits per race can be built with *short* encodings for *more probable* horses, and *longer* encodings for *less probable* horses. E.g. we could encode the most likely horse with the code `0`, and the remaining horses as `10`, then `110`, `1110`, `111100`, `111101`, `111110`, and `111111`.

Suppose horses are equally likely. In this case each horse would have a probability of $\frac{1}{8}$. The entropy is then
$$
H(X)=-\sum\_{i=1}^{i=8} \frac{1}{8} \log \frac{1}{8}=-\log \frac{1}{8}=3 \mathrm{bits}
$$

------

Most of what we will use entropy for involves ***sequences***.

For a grammar, for example, we will be computing the entropy of some sequence of words $W=\{w\_0, w\_1, w\_2, \dots, w\_n\}$. One way to do this is to have a variable that ranges over sequences of words. For example we can compute the entropy of a random variable that ranges over all ***finite*** sequences of words of length $n$ in some language $L$
$$
H\left(w\_{1}, w\_{2}, \ldots, w\_{n}\right)=-\sum_{W\_{1}^{n} \in L} p\left(W\_{1}^{n}\right) \log p\left(W\_{1}^{n}\right)
$$
**Entropy rate** (**per-word entropy**): entropy of this sequence divided by the number of word
$$
\frac{1}{n} H\left(W\_{1}^{n}\right)=-\frac{1}{n} \sum_{W\_{1}^{n} \in L} p\left(W\_{1}^{n}\right) \log p\left(W\_{1}^{n}\right)
$$
For sequence $L$ of ***infinite*** length, the entropy rate $H(L)$ is
$$
\begin{aligned}
H(L) &=\lim \_{n \rightarrow \infty} \frac{1}{n} H\left(w\_{1}, w\_{2}, \ldots, w\_{n}\right) \\\\
&=-\lim \_{n \rightarrow \infty} \frac{1}{n} \sum_{W \in L} p\left(w\_{1}, \ldots, w_{n}\right) \log p\left(w\_{1}, \ldots, w\_{n}\right)
\end{aligned}
$$

## The Shannon-McMillan-Breiman theorem

If the language is regular in certain ways (to be exact, if it is both **stationary** and **ergodic**), then
$$
H(L)=\lim \_{n \rightarrow \infty}-\frac{1}{n} \log p\left(w\_{1} w\_{2} \ldots w\_{n}\right)
$$
I.e., we can take a single sequence that is long enough instead of summing over all possible sequences. 

- 💡 Intuition: a long-enough sequence of words will contain in it many other shorter sequences and that each of these shorter sequences will reoccur in the longer sequence according to their probabilities.

**Stationary**

A stochastic process is said to be **stationary** if the probabilities it assigns to a sequence are *invariant* with respect to shifts in the time index.

- I.e., the probability distribution for words at time $t$ is the same as the probability distribution at time $t+1$.
- Markov models, and hence n-grams, are stationary.
  - E.g.,  in bigram, $P_i$ is dependent only on $P_{i-1}$. If we shift our time index by $x$, $P_{i+x}$ is still dependent on  $P_{i+x-1}$

- Natural language is NOT stationary
  - the probability of upcoming words can be dependent on events that were arbitrarily distant and time dependent. 

To summarize, by making some incorrect but convenient simplifying assumptions, **we can compute the entropy of some stochastic process by taking a very long sample of the output and computing its average log probability.**

## Cross-entropy

Useful when we don’t know the actual probability distribution $p$ that generated some data

It allows us to use some $m$, which is a model of $p$ (i.e., an approximation to $p$). The

cross-entropy of $m$ on $p$ is defined by


$$
H(p, m)=\lim\_{n \rightarrow \infty}-\frac{1}{n} \sum\_{W \in L} p\left(w\_{1}, \ldots, w\_{n}\right) \log m\left(w\_{1}, \ldots, w\_{n}\right)
$$


(we draw sequences according to the probability distribution $p$, but sum the log of their probabilities according to $m$)

Following the Shannon-McMillan-Breiman theorem, for a stationary ergodic process: 
$$
H(p, m)=\lim\_{n \rightarrow \infty}-\frac{1}{n} \log m\left(w\_{1} w\_{2} \ldots w\_{n}\right)
$$
(as for entropy, we can estimate the cross-entropy of a model $m$ on some distribution $p$ by taking a single sequence that is long enough instead of summing over all possible sequences)

The cross-entropy $H(p. m)$ is an **upper bound** on the entropy $H(p)$:
$$
H(p)\leq H(p, m)
$$
This means that we can use some simplified model $m$ to help estimate the true entropy of a sequence of symbols drawn according to probability $p$

- The more accurate $m$ is, the closer the cross-entropy $H(p, m)$ will be to the true entropy $H(p)$
  - Difference between $H(p, m)$ and $H(p)$ is a measure of how accurate a model is
  - The more accurate model will be the one with the lower cross-entropy. 

### Relationship between perplexity and cross-entropy

Cross-entropy is defined in the limit, as the length of the observed word sequence goes to infinity. We will need an approximation to cross-entropy, relying on a (sufficiently long) sequence of fixed length. 

This approximation to the cross-entropy of a model $M=P\left(w_{i} | w_{i-N+1} \dots w_{i-1}\right)$ on a sequence of words $W$ is
$$
H(W)=-\frac{1}{N} \log P\left(w_{1} w_{2} \ldots w_{N}\right)
$$
The **perplexity** of a model $P$ on a sequence of words $W$ is now formally defined as

the exp of this cross-entropy:
$$
\begin{aligned}
\operatorname{Perplexity}(W) &=2^{H(W)} \\\\
&=P\left(w_{1} w_{2} \ldots w_{N}\right)^{-\frac{1}{N}} \\\\
&=\sqrt[N]{\frac{1}{P\left(w_{1} w_{2} \ldots w_{N}\right)}} \\\\
&=\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P\left(w_{i} | w_{1} \ldots w_{i-1}\right)}}
\end{aligned}
$$
