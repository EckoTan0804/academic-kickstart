---
# Title, summary, and position in the list
linktitle: "04-NER"
summary: ""
weight: 2050

# Basic metadata
title: "Name Entity Recognition"
date: 2020-09-15
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "Lecture Notes"]
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
        parent: lecture-notes
        weight: 5

---

## Introduction

### Definition

**Named Entity**: some entity represented by a name

**Named Entity Recognition**: Find and classify named entities in text

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-26%2018.13.30.png" alt="截屏2020-05-26 18.13.30" style="zoom: 40%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-26%2018.13.49.png" alt="截屏2020-05-26 18.13.49" style="zoom:40%;" />

### Why useful?

- Create indices & hyperlinks

  

- Information extraction
  
  - Establish relationships between named entities, build knowledge base
  
- Question answering: answers often NEs

- Machine translation: NEs require special care
  
  - NEs often unknown words, usually passed through without translation.

### Why difficult?

- World knowledge
- Non-local decisions

- Domain specificity

- Labeled data is very expensive



## Label Representation

- **IO** 

  - **I**: Inside
  - **O**: Outside (indicates that a token belongs to no chunk)

- **BIO**

  - **B**: Begin

    > [Wiki](https://en.wikipedia.org/wiki/Inside–outside–beginning\_(tagging))
    >
    > The **IOB format** (short for inside, outside, beginning), a synonym for **BIO format**,  is a common tagging format for tagging [tokens](https://en.wikipedia.org/wiki/Lexical\_token) in a [chunking](https://en.wikipedia.org/wiki/Chunking\_(computational\_linguistics)) task in [computational linguistics](https://en.wikipedia.org/wiki/Computational\_linguistics).
    >
    > - **B**-prefix before a tag: indicates that the tag is the **beginning** of a chunk
    > - **I**-prefix before a tag: indicates that the tag is the **inside** of a chunk
    > - **O** tag: indicates that a token belongs to NO chunk

- **BIOES**

  - **E**: Ending character
  - **S**: single element

- **BILOU**

  - **L**: Last character
  - **U**: Unit length

  

Example:

```tex
Fred showed Sue Mengqiu Huang's new painting
```



<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-26%2018.19.51.png" alt="截屏2020-05-26 18.19.51" style="zoom:50%;" />

### Data

- CoNLL03 shared task data
- MUC7 dataset
- Guideline examples for special cases:
  - Tokenization
  - Elision

## Evaluation

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-26%2022.46.58.png" alt="截屏2020-05-26 22.46.58" style="zoom:50%;" />

- **Precision and Recall**
  $$
  \text{Precision} = \frac{\\# \text { correct labels }}{\\# \text { hypothesized labels }} = \frac{TP}{TP + FP}
  $$

  $$
  \text{Recall} = \frac{\\# \text { correct labels }}{\\# \text { reference labels }} = \frac{TP}{TP + FN}
  $$


  - Phrase-level counting

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-26%2022.47.59.png" alt="截屏2020-05-26 22.47.59" style="zoom:40%;" />

    > - System 1:
    >
    >   - "$\text{\\$200,000,000}$" is correctly recognized as NE $\Rightarrow$ TP =1
    >   - "First Bank of Chicago" is incorrectly recognised as non-NE (i.e., O) $\Rightarrow$ FN = 1
    >
    >   Therefore:
    >
    >   - $\text{Precision} = \frac{1}{1 + 0} = 1$
    >
    >     - $\text{Recall} = \frac{1}{1 + 1} = \frac{1}{2}$ 
    >
    > - System 2:
    >
    >   - "$\text{\\$200,000,000}$" is correctly recognized as NE $\Rightarrow$ TP =1
    >
    >   - For "First Bank of Chicago" 
    >
    >     | Word            | Actual label | Predicted label |
    >     | --------------- | ------------ | --------------- |
    >     | First           | ORG          | O               |
    >     | Bank of Chicago | ORG          | ORG             |
    >
    >     There's a boundary error (since we consider the whole phrase):
    >
    >     - FN = 1
    >     - FP = 1
    >
    >   Therefore:
    >
    >   - $\text{Precision} = \frac{1}{1 + 1} = \frac{1}{2}$
    >
    >   - $\text{Recall} = \frac{1}{1 + 1} = \frac{1}{2}$ 

    - Problems
      - Punish partial overlaps
      - Ignore true negatives

  - Token-level

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-26%2023.51.18.png" alt="截屏2020-05-26 23.51.18" style="zoom: 67%;" />

    > In token-level, we consider these tokens: "First", "Bank", "of", "Chicago", and "\$200,000,000"
    >
    > - System 1
    >
    >   - "$\text{\\$200,000,000}$" is correctly recognized as NE $\Rightarrow$ TP =1
    >   - "First", "Bank", "of", "Chicago" are incorrectly recognised as non-NE (i.e., O) $\Rightarrow$ FN = 4
    >
    >   Therefore:
    >
    >   - $\text{Precision} = \frac{1}{1 + 0} = 1$
    >
    >   - $\text{Recall} = \frac{1}{1 + 4} = \frac{1}{5}$ 

    - Partial overlaps rewarded!
    - But
      - longer entities weighted more strongly 
      
        
      
      - True negatives still ignored 🤪

- **$F\_1$ score** (harmonic mean of precision and recall)
  $$
  F\_1 = \frac{2 \times \text { precision } \times \text { recall }}{\text { precision }+\text { recall }}
  $$

## Text Representation

### Local features

- Previous two predictions (tri-gram feature)

  - $y\_{i-1}$ and $y\_{i-2}$

- Current word $x\_i$

- Word type $x\_i$

  - all-capitalized, is-capitalized, all-digits, alphanumeric, ...

- Word shape

  - lower case - 'x'

  - upper case - 'X'

  - numbers - 'd'

  - retain punctuation

    ![截屏2020-05-27 10.18.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-27%2010.18.48-20200915221013059.png)

- Word substrings

- Tokens in window

- Word shapes in window

- ...

### Non-local features

**Identify tokens that should have same labels**

Type:

- **Context aggregation**
  
  - Derived from all words in the document
  
    
  
  - No dependencies on predictions, usable with any inference algorithm
- **Prediction aggregation**
  
  - Derived from predictions of the whole document
  
    
  
  - Global dependencies; Inference:
    - first apply baseline without non-local features
    
      
    
    - then apply second system conditioned on output of first system
- **Extended prediction history**
  
  - Condition only on past predictions --> greedy / beam search
  
    
  
  - 💡 Intuition: Beginning of document often easier, later in document terms often get abbreviated

## Sequence Model

### HMMs

- Generative model

- Generative story:

  - Choose document length $N$

  - For each word $t = 0, \dots, N$:

    - Draw NE label $\sim P(y\_t | y\_{t-1})$

    - Draw word $\sim P\left(x\_{t} | y\_{t}\right)$
      $$
      P(\mathbf{y}, \mathbf{x})=\prod P\left(y\_{t} | y\_{t-1}\right) P\left(x | y\_{t}\right)
      $$


- Example
  
  ![截屏2020-09-15 22.19.02](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-15%2022.19.02.png)
  
- 👍 Pros
  
  - intuitive model
  - Works with unknown label sequences 
  - Fast inference
  
- 👎 Cons
  - Strong limitation on textual features (conditional independence)
  - Model overly simplistic (can improve the generative story but would lose fast inference)

### Max. Entropy

- Discriminative model $P(y\_t|x)$
- Don’t care about generation process or input distribution
- Only model conditional output probabilities
- 👍 Pros: Flexible feature design

- 👎 Cons: local classifier -> disregard sequence information

### CRF

- Discriminative model
- 👍 Pros:
  - Flexible feature design

  - Condition on local sequence context 
  - Training as easy as MaxEnt

- 👎 Cons: Still no long-range dependencies possible



## Modelilng

Difference to POS

- Long-range dependencies
- Alternative resources can be very helpful
- Several NER more than one word long

Example

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-27%2010.36.36.png" alt="截屏2020-05-27 10.36.36"  />

## Inference

### Viterbi

- Finds exact solution
- Efficient algorithmic solution using dynamic programming
- Complexity exponential in order of Markov model 
  - Only feasible for small order

### Greedy

- At each timestep, choose locally best label

- Fast, support conditioning on global history (not future) 
- No possibility for “label revision”

### Beam

- Keep a beam of the $n$ best greedy solutions, expand and prune 
- Limited room for label revisions

### Gibbs Sampling

- Stochastic method

- Easy way to sample from multivariate distribution

- Normally used to approximate joint distributions or intervals
  $$
  P\left(y^{(t)} | y^{(t-1)}\right)=P\left(y\_{i}^{(t)} | y\_{-i}^{(t-1)}, x\right)
  $$

  - $-1$ means all states except $i$

- 💡 Intuitively:

  - Sample one variable at a time, conditioned on current assignment of all other variables
  - Keep checkpoints (e.g. after each sweep through all variables) to approximate distribution

- In our case:

  - Initialize NER tags (e.g. random or via Viterbi baseline model) 

  - Re-sample one tag at a time, conditioned on input and all other tags

  - After sampling for a long time, we can estimate the joint distribution over outputs $P(y|x)$

  - However, it’s slow, and we may only be interested in the best output 🤪

  - Could choose **best instead of sampling**
    $$
    y^{(t)}=y\_{-i}^{(t-1)} \cup \underset{y\_{i}^{(t)}}{\operatorname{argmax}}\left(P\left(y\_{i}^{(t)} | y\_{-i}^{(t-1)}, x\right)\right)
    $$

    - will get stuck in local optima 😭

  - Better: **Simulated annealing** 

    - Gradually move from sampling to argmax
      $$
      P\left(y^{(t)} | y^{(t-1)}\right)=\frac{P\left(y\_{i}^{(t)} | y\_{-i}^{(t-1)}, x\right)^{1 / c\_{t}}}{\displaystyle\sum\_{j} P\left(y\_{j}^{(t)} | y\_{-j}^{(t-1)}, x\right)^{1 / c\_{t}}}
      $$

## External Knowledge

### Data

- Supervised learning:
  - Label Data: 
    - Text
    - NE Annotation

- Unsupervised learning
  - Unlabeled Data: Text
  - Problem: Hard to directly learn NER
- Semi-supervised: Labeled and Unlabeled Data

### Word Clustering

- Problem: Data Sparsity
- Idea
  - Find lower-dimensional representation of words

  - real vector /probabilities have natural measure of similarity

- Which words are similarr?
  - Distributional notion
  - if they appear in similar context, e.g.
    -  “president” and “chairman” are similar 
    -  “cut” and “knife” not

**Words in same cluster should be similar**

### Brown clusters

- **Bottom-up** agglomerative word clustering

- Input: Sequence of words $w\_1, \dots, w\_n$

- Output

  - binary tree
  - Cluster: subtree (according to desired #clusters)

- 💡 Intuition: put syntacticly "exchangable" words in same cluster. E.g.:

  - Similar words: president/chairman, Saturday/Monday

  - Not similar: cut/knife

- Algorithm:

  - Initialization: Every word is its own cluster

  - While there are more than one cluster
    - Merge two clusters that maximizes the quality of the clustering

- Result:

  - **Hard** clustering: each word belongs to **exactly one** cluster

- Quality of the clustering

  - Use class-based bigram language model

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-27%2011.09.55-20200916110552353.png" alt="截屏2020-05-27 11.09.55" style="zoom:150%;" />

  - Quality: logarithm of the probability of the training text normalized by the length of the text
    $$
    \begin{aligned}
    \text { Quality }(C) &=\frac{1}{n} \log P\left(w\_{1}, \ldots, w\_{n}\right) \\\\
    &=\frac{1}{n} \log P\left(w\_{1}, \ldots, w\_{n}, C\left(w\_{1}\right), \ldots, C\left(w\_{n}\right)\right) \\\\
    &=\frac{1}{n} \log \prod\_{i=1}^{n} P\left(C\left(w\_{i}\right) | C\left(w\_{i-1}\right)\right) P\left(w\_{i} | C\left(w\_{i}\right)\right)
    \end{aligned}
    $$

  - Parameters: estimated using maximum-likelihood

