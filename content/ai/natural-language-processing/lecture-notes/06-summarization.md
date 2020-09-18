---
# Title, summary, and position in the list
linktitle: "06-Summarization"
summary: ""
weight: 2070

# Basic metadata
title: "06-Summarization"
date: 2020-09-16
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "Lecture Notes"]
categories: ["NLP"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading\\_time: true # Show estimated reading time?
summary: ""
share: false  # Show social sharing links?
featured: true

comments: false  # Show comments?
disable\\_comment: true
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
        weight: 7

---

## TL;DR

- [Text summarization](#what-is-summarization)

- Most important technique

  - [Extraction](#extraction) 

- Tasks:

  - [Key word extraction](#key-word-extraction)
  - [Sentence extraction](#sentence-extraction)

- Algorithms:

  - [Supervised](#supervised-approaches)

  - [Unsupervised](#unsupervised-approaches)

- Abstract summarization still an open problem

![截屏2020-09-16 23.43.02](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2023.43.02.png)

## Introduction

### **What is Summarization?**

- Reduce natural language text document
- Goal: **Compress** text by extracting the most important/relevant parts :muscle:

### Applications

- Articles, news: Outlines or abstracts
- Email / Email threads

- Health information

- Meeting summarization

### Dimensions

- **Single vs. multiple**
  - **Single-document** summarization
    - Given single document
    - Produce abstract, outline, headline, etc.
  - **Multiple-document** summarization
    - Given a group of documents

    - A series of news stories on the same event

    - A set of web pages about some topic or question

- **Generic vs. Query-focused summarization**
  - **Generic** summarization: summarize the content of the document
  - **Query-focused** summarization: kind of *complex* question answering 🤪
    - Summarize a document with respect to an information need expressed in a user query
    - Longer, descriptive, more informative answers
    - Answer a question by summarizing a document that has information to construct the answer

## Techniques

### Extraction

- **Select subset of existing text segments**
- e.g.:
  - Sentence extraction

  - Key-phrase extraction
- Simpler, most focus in research

### Abstraction

- Use natural language generation to create summary 
- More human like

## Extractive summarization

Three main components

- Content selection ("*Which parts are important to be in the summary?*")
- Information ordering ("*How to order summaries?*")
- Sentence realization (Clean up/Simplify sentences)

 ### Supervised approaches 

#### Key-word extraction

- Given: Text (e.g. abstract of an article, ...)

- **Task: Find most important key phrases**

  | Computer                         | Human                   |
  | -------------------------------- | ----------------------- |
  | Select key phrases from the text | Abstraction of the text |
  | No new wordings                  | New words               |

##### **Key-phrase extraction using Supervised approaches**

- Given: Collection of text with key-words

- Algorithm

  - Extract all uni-grams, bi-grams and tri-grams

    - Example

      ![截屏2020-09-17 23.08.38](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-17%2023.08.38.png)
      
      Extraction: Compatibility, Compatibility of, Compatibility of systems, of, of systems, of systems of, systems, systems of, systems of linear, of linear, of linear constraints, linear, linear constraints, linear constraints over, ...

  - Annotate each examples with features

  - Annotate training examples with class:

    - 1 if sequence is part of the key words
    - 0 if sequence is not part of the key words

  - Train classifier

  - Create test examples and classify

- Examples set

  - All uni-, bi-, and trigrams (except punctuation) 
  - restrict to certain POS sets

- 🔴 Problem:

  - Enough examples to generate all/most key phrases

  - Too many examples -> low performance of classifier

- Features

  - Term frequency

  - TF-IDF (Term frequency–inverse document frequency)

    - Reflect importance of a word in a document
      $$
      \text{TF-IDF} = tf * idf
      $$

      - $tf(w, D)$: Number of occurrences of word $w$ in document $d$ divided by the maximum frequency of one word in $D$
        $$
        t f(w, D)=\\#(w, D) \frac{\\#(w, D)}{\max_{w^{\prime} \in D}\left(w^{\prime}, D\right)}
        $$

      - $idf(w)$: Number of documents divided by the number of documents

        which contain $w$
        $$
        i d f(w)=\log \frac{|D|}{|\\{d \in D: w \in d\\}|}
        $$

  - Length of the example

  - Relative position of the first occurrence

  - Boolean syntactic features 

    - contains all caps

- Learning algorithm

  - Decision trees,

  - Naive Bayes classifier 
  - ...

- Evaluation

  - Compare results to reference

    - Test set: Text

    - Human generated Key words

  - Metrics:

    - Precision 
    - Recall

    - F-Score

  - 🔴 Problems

    - Humans do not only extract key words, but also abstract 
    - Normally not all key words are reachable

####Sentence extraction

- Use statistic heuristics to select sentences 

- Do not change content and meaning

- **💡 Idea**

  - Use measure to determine importance of sentence
    - TF-IDF
    - Supervised trained combination of several features
  - Rank sentence according to metric
  - Output sentences with highest scores: 
    - Fixed number
    - All sentence above threshold

- Limitations: Do NOT change text (e.g. add phrases, delete parts of the text)

- Evaluation

  - Idea: Compare automatic summary to abstract of text

    - <span style="color:red">Problem: Different sentences --> Nearly no exact match</span> 😭

  - **ROUGE - Recall-Oriented Understudy for Gisting Evaluation**

    - Use also approximate matches

    - Compare automatic summary to human generated text

      - Given a document D and an automatic summary X

        - M humans produce a set of reference summaries of D

        - What percentage of the n-grams from the reference summaries appear in X?

    - ROUGE-N: Overlap of N-grams between the system and reference summaries
      $$
      \text{ROUGH-N} = \frac{\sum_{S \in \\{\text{Reference Summaries}\\}} \sum_{gram_n \in S}\operatorname{Count}_{match}(gram_n)}{\sum_{S \in \\{\text{Reference Summaries}\\}} \sum_{gram_n \in S}\operatorname{Count}(gram_n)}
      $$

      - Example: 

        Auto-generated summary ($Y$)

        ```tex
        the cat was found under the bed
        ```

        Gold standard (human produced) ($X1$)

        ```tex
        the cat was under the bed
        ```

        1-gram and 2-gram summary:

        | #     | 1-gram                                 | reference 1-gram                       | 2-gram                                     | reference 2-gram                           |
        | ----- | -------------------------------------- | -------------------------------------- | ------------------------------------------ | ------------------------------------------ |
        | 1     | <span style="color:green">the</span>   | <span style="color:green">the</span>   | <span style="color:green">the cat</span>   | <span style="color:green">the cat</span>   |
        | 2     | <span style="color:green">cat</span>   | <span style="color:green">cat</span>   | <span style="color:green">cat was</span>   | <span style="color:green">cat was</span>   |
        | 3     | <span style="color:green">was</span>   | <span style="color:green">was</span>   | was found                                  | was under                                  |
        | 4     | found                                  | <span style="color:green">under</span> | found under                                | <span style="color:green">under the</span> |
        | 5     | <span style="color:green">under</span> | <span style="color:green">the</span>   | <span style="color:green">under the</span> | <span style="color:green">the bed</span>   |
        | 6     | <span style="color:green">the</span>   | <span style="color:green">bed</span>   | <span style="color:green">the bed</span>   |                                            |
        | 7     | <span style="color:green">bed</span>   |                                        |                                            |                                            |
        | count | 7                                      | 6                                      | 6                                          | 5                                          |

        - $\operatorname{ROUGE}-1(X1, Y) = \frac{6}{6} = 1$
        - $\operatorname{ROUGE}-2(X1, Y) = \frac{4}{5}$

 ### Unsupervised approaches 

Problems of supervised approaches: <span style="color:red">Hard to acquire training data</span>

We try to use unsupervised learning to find key phrases / sentences which are most important. But which sentences are most important?

💡 Idea: Sentences which are most similar to the other sentences in the text

### Graph-based approaches

- Map text into a graph 
  - Nodes: 
    - Text segments: Words
  - Edges: Similarity
- Find most important/central vertices
- Algorithms: TextRank / LexRank

#### Graph-based approaches : **Key-phrase extraction**

- Graph

  - Nodes
    - Text segments: Words
    - Restriction:
      - Nouns
      - Adjective
  - Edges
    - items co-occur in a window of N words

- Calculate most important nodes

- Build Multi-word expression in **post-processing**

  - Mark selected items in original text 
  - If two adjacent words are marked --> Collapse to one multi-words expression

- Example

  ![截屏2020-09-18 00.43.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2000.43.48.png)

#### **Graph-based approaches : Sentence extraction**

Graph:

- Nodes: Sentences 

- Edges: Fully connected with weights 

- Weights:
  - TextRank: Word overlap normalized to sentence length
    $$
    \text {Similarity}\left(S_{i}, S_{j}\right)=\frac{\left|\left\\{w_{k} \mid w_{k} \in S_{i} \& w_{k} \in S_{j}\right\\}\right|}{\log \left(\left|S_{i}\right|\right)+\log \left(\left|S_{j}\right|\right)}
    $$

  - LexRank: Cosine Similarity of TF-IDF vectors
    $$
    \text { idf-modified-cosine }(x, y)=\frac{\sum_{w \in x, y} \mathrm{tf}_{w, x} \mathrm{tf}_{w, y}\left(\mathrm{idf}_{w}\right)^{2}}{\sqrt{\sum_{x_{i} \in x}\left(\mathrm{tf}_{x_{i}, x} \mathrm{idf}_{x_{i}}\right)^{2}} \times \sqrt{\sum_{y_{i} \in y}\left(\mathrm{tf}_{y_{i}, y} \mathrm{idf}_{y_{i}}\right)^{2}}}
    $$



## Abstract summarization

- Sequence to Sequence task 
  - Input: Document 
  - Output: Summary
- Several NLP tasks can be modeled like this (ASR, MT,...)
- Successful deep learning approach: **Encoder-Decoder Model**

### Sequence-to-Sequence Model

<img src="https://miro.medium.com/max/3972/1*1JcHGUU7rFgtXC_mydUA_Q.jpeg" alt="Understanding Encoder-Decoder Sequence to Sequence Model | by Simeon  Kostadinov | Towards Data Science" style="zoom: 33%;" />

- Predict words based on 
  - previous target words and 
  - source sentence
- Encoder: Read in source sentence
- Decoder: Generate target sentence word by word

#### Encoder

![截屏2020-09-18 10.11.37](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2010.11.37.png)

- Read in input: Represent content as hidden vector with fixed dimension 

- LSTM-based model

- Fixed-size sentence representation

- Details:

  ![截屏2020-09-18 10.12.59](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2010.12.59.png)

  - One–hot encoding 
  - Word embedding 
  - RNN layer(s)

#### Decoder

![截屏2020-09-18 10.13.33](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2010.13.33.png)

- Generate output: Use output of encoder as input
- LSTM-based model

- Input last target word

### Attention-based Encoder-Decoder

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2010.17.42.png" alt="截屏2020-09-18 10.17.42" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2010.16.48.png" alt="截屏2020-09-18 10.16.48" style="zoom:67%;" />

#### Attention-based Encoder : copy mechanism

Calculate probability “**better to generate one word from vocabulary than to copy a word from source sentence**“
$$
p_{g e n}=\sigma\left(w_{c}^{T} c_{t}+w_{s}^{T} s_{t}+w_{x}^{T} x_{t}+b_{p t r}\right)
$$
Word with the **highest probability** should be the output word
$$
P(w)=p_{g e n} P_{v o c a b}(w)+\left(1-p_{g e n}\right) \sum_{j: w_{j}=w} \alpha_{i j}
$$

### Data

- Training data

  - Documents and summary 

- DUC data set

  - News article

  - Around 14 word summary

-  Giga word

  - News articles

  - Headline generation 

- CNN/Mail Corpus

  - Article

  - Predict bullet points