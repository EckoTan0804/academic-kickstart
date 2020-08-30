---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 420

# Basic metadata
title: "Train Naive Bayes Classifiers"
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
        weight: 2

---

## Maximum Likelihood Estimate (MLE)

In Naive Bayes calculation we have to learn the probabilities $P(c)$ and $P(w_i|c)$. We use the **Maximum Likelihood Estimate (MLE)** to estimate them. We’ll simply use the **frequencies** in the data.

- $P(c)$: document prior

  - "what percentage of the documents in our training set are in each class $c$ ?"

  $$
  \hat{P}(c)=\frac{N_{c}}{N_{d o c}}
  $$

  - $N_c$: the number of documents in our training data with class $c$
  - $N_{doc}$: the total number of documents

- $P(w_i|c)$

  - "The fraction of times the word $w_i$ appears among all words in all documents of topic $c$"

    - We first concatenate all documents with category $c$ into one big “category $c$” text.

    - Then we use the frequency of $w_i$ in this concatenated document to give a maximum likelihood estimate of the probability
      $$
      \hat{P}\left(w_{i} | c\right)=\frac{\operatorname{count}\left(w_{i}, c\right)}{\sum_{w \in V} \operatorname{count}(w, c)}
      $$

      - $V$: vocabulary that consists of the union of all the word types in all classes, not just the words in one class $c$

  - Avoid zero probablities in the likelihood term for any class: **Laplace (add-one) smoothing**
    $$
    \hat{P}\left(w_{i} | c\right)=\frac{\operatorname{count}\left(w_{i}, c\right)+1}{\sum_{w \in V}(\operatorname{count}(w, c)+1)}=\frac{\operatorname{count}\left(w_{i}, c\right)+1}{\left(\sum_{w \in V} \operatorname{count}(w, c)\right)+|V|}
    $$

    > Why is this a problem?
    >
    > Imagine we are trying to estimate the likelihood of the word “fantastic” given class *positive*, but suppose there are no training documents that both contain the word “fantastic” and are classified as *positive*. Perhaps the word “fantastic” happens to occur (sarcastically?) in the class *negative*. In such a case the probability for this feature will be zero:
    > $$
    > \hat{P}(\text { "fantastic" } | \text { positive })=\frac{\text { count }(\text { "fantastic}" \text {, positive })}{\sum_{w \in V} \text { count }(w, \text { positive })}=0
    > $$
    > But since naive Bayes naively multiplies all the feature likelihoods together, zero probabilities in the likelihood term for any class will cause the probability of the class to be zero, no matter the other evidence!

In addition to $P(c)$ and $P(w_i|c)$, We should also deal with

- **Unknown words** 
  - Words that occur in our test data but are NOT in our vocabulary at all because they did not occur in any training document in any class
  - 🔧 Solution: Ignore them
    - Remove them from the test document and not include any probability for them at all
- **Stop words**
  - Very frequent words like *the* and *a*
  - Solution:
    - Method 1: 
      1. sorting the vocabulary by frequency in the training set
      2. defining the top 10–100 vocabulary entries as stop words
    - Method 2: 
      1. use one of the many pre-defined stop word list available online
      2. Then every instance of these stop words are simply removed from both training and test documents as if they had never occurred.
  - In most text classification applications, however, using a stop word list does NOT improve performance 🤪, and so it is more common to make use of the entire vocabulary and not use a stop word list.

### Example

We’ll use a sentiment analysis domain with the two classes positive (+) and negative (-), and take the following miniature training and test documents simplified from actual movie reviews.

![截屏2020-06-14 12.37.43](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-14%2012.37.43-20200803120410651.png)

- The prior $P(c)$:
  $$
  P(-)=\frac{3}{5} \quad P(+)=\frac{2}{5}
  $$

The word *with* doesn’t occur in the training set, so we drop it completely.

The remaining three words are "predictable", "no", and "fun". Their likelihoods from the training set are (with Laplace smoothing): 
$$
\begin{aligned}
P\left(\text { "predictable" }|-\right) &=\frac{1+1}{14+20} \qquad P\left(\text { "predictable" } |+\right)=\frac{0+1}{9+20} \\\\
P\left(\text{"no"}|-\right) &=\frac{1+1}{14+20} \qquad P\left(\text{"no"}|+\right)=\frac{0+1}{9+20} \\\\
P\left(\text{"fun"}|-\right) &=\frac{0+1}{14+20} \qquad P\left(\text{"fun"}|+\right)=\frac{1+1}{9+20}
\end{aligned}
$$

> $$
> \begin{aligned}
> V=\\{
> \& \text{just, plain, boring, entirely, predictable, and, lacks, energy, no, suprises, very, few, laughs, } \\\\
> \& \text{powerful, the, most, fun, film, of, summer}
> \\}
> \end{aligned}
> $$
>
> $\Rightarrow |V|=20$
>
> The word "predictable" occurs in negative (-) once, therefore, with Laplace smoothing:
>
> $P\left(\text { "predictable" }|-\right) =\frac{1+1}{14+20}$

For the test sentence $S=$ "predictable with no fun", after removing the word "with":
$$
\begin{array}{l}
P(-) P(S |-)=\frac{3}{5} \times \frac{2 \times 2 \times 1}{34^{3}}=6.1 \times 10^{-5} \\\\
P(+) P(S |+)=\frac{2}{5} \times \frac{1 \times 1 \times 2}{29^{3}}=3.2 \times 10^{-5}
\end{array}
$$
$P(-)P(S|-) > P(+)P(S|+)$ 

$\Rightarrow$ The model predicts the class *negative* for the test sentence $S$.

