---
# Title, summary, and position in the list
linktitle: "07-QA"
summary: ""
weight: 2080

# Basic metadata
title: "Question Answering"
date: 2020-09-18
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
        weight: 8

---



![截屏2020-09-18 10.32.49](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2010.32.49.png)

## Definition

**Question Answering**

- Automatically answer questions posed by humans in natural language 
- Give user short answer to their question

- Gather and consult necessary information

Related topics

- Information Retrieval 
- Reading Comprehension 
- Database Access

- Dialog

- Text Summarization

## Problem Dimensions 

### Questions 

- **Question class**
  - Almost universally factoid questions
     E.g.: *“What does the Peugeot company manufacture?”*
  - More open in dialog context
- **Question domain**
  - Topic of the content

  - Open-Domain: Any topic

  - Closed-Domain: Specific topic, e.g. movies, sports, etc
- **Context**
  - How much context is provided? 
  - Is search necessary?
- **Answer types**
  - Factual Answers 
  - Opinion 
  - Summary
- **Kind of questions**
  - Yes/No
  - “wh”-questions

  - Indirect requests (I would like to...) 
  - Commands

### Applications

- **Knowledge source types**
  - Structured data (database)

  - Semi-structured data (e.g. Wikipedia tables) 
  - Free text (e.g. Wikipedia text)
- **Knowledge source origins** 
  - Search over the web 
  - Search of a collection 
  - Single text
- **Domain** 
  - Domain-independent 
  - Domain-specific system

### Users

- **First time/casual users** 
  - Explain limitations

- **Power users**
  - Emphasize novel information

  - Omit previously provided information

### Answers

- Long 
- Short 
- Lists 
- Narrative 
- Creation
  - Extraction 
  - Generation

### Evaluation

- **What is a good answer?**
- **Should the answer be short or long?**
  - Easier to have the answer in longer segments 
  - Less concise, more comprehensive

### Presentation

- Underspecified question 

  - Feedback

  - Too many documents

- Text or speech input

## Examples

- **TREC**
- **SQuAD** (Stanford Question Answering Dataset)
- **IBM Watson**

## Motivation

- Vast amounts of information written by humans for humans 
- Computers are good at searching vast amounts of information 
- Natural interaction with computers :muscle:

## System Approaches

### Text-based system

- Use **information retrieval** to search for matching documents

### Knowledge-based approaches

- Build **semantic representation** of the query

- Retrieve answer from semantic databases (Ontologies)

### Knowledge-rich / hybrid approaches

Combine both

## QA System Overview 

### Components

- **Information Retrieval**
  - Need to find good text segments

- **Answer Extraction**
  - Given some context and the question, produce an answer
  - Either part may be supplemented by other NLP tools

**Common Components**

![截屏2020-09-18 13.12.43](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2013.12.43.png)

### Preprocessing 

### Question Analysis 

![截屏2020-09-18 13.02.34](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2013.02.34.png)

- **Input: Natural language question**

  - Implicit input 

    - Dialog state

    - User information

  - Derived inputs

    - POS-tags, NER, dependency graph, syntax tree, etc.

- **Output: Representation for Information Retrieval and Answer Extraction**

  - For **IR**: Weighted vector or search term collection
  - For **answer extraction**
    - Lexical answer type (person/company/acronym/...) 
    - Additional constraints (e.g. relations)

#### Answer Type Classification

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2013.04.13.png" alt="截屏2020-09-18 13.04.13" style="zoom:80%;" />

- **Classical approach**: Question word (who, what, where,...) 

  - When: date

  - Who: person 
  - Where: location

- Examples

  - Regular expressions

    Who {is | was | are | were } – Person

  - Question head word (First noun phrase after the question word)
    - Which **city** in China has the largest number of foreign financial companies? 
    - What is the state **flower** of California?

- 🔴 Problems

  - “Who” questions could refer to e.g. companies
    - E.g. *"Who makes the Beetle?"*
  - Which / What is not clear
    - E.g. *"What was the Beatles’ first hit single?"*

- Approaches

  - Manually created question type hierarchy 
  - Machine learning classification

  (Current ML systems often do NOT use Answer Type Classification 😂)

#### Constraints

- Keyword extraction

  - Expand keywords using synonyms

- Statistical parsing

  - Identify semantic constraints

- Example

  Represent a question as bag-of-words

  - *“What was the monetary value of the Nobel Peace Price in 1989?”* 

    ```monetary, value, Nobel, Peace, Price, 1989```

  - *“What does the Peugeot company manufacture?”*

    ```Peugeot, company, manufacture```

  - *“How much did Mercury spend on advertising in 1993?”*

    ```Mercury, spend, advertising, 1993```

### Retrieval: Candidate Document Selection

![截屏2020-09-18 13.10.29](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2013.10.29.png)

- Most common approach:

  - **Conventional Information Retrieval search**
    - Using search indices 
    - Lucene

    - TF-IDF
  - **Several stages**: Coarse-to-fine search

- Result: Small set of documents for detailed analysis

- Decisions: Boolean vs. rank-based engines

- Retrieve only part of the document

  - Mostly only part of the document is important 

- Passage retrieval

  - Return only **subsets** of the document

- Segment document into coherent text segments

- Combine results from multiple search engines

- **Text-based system**

  - Use only syntactic information such as n-grams

  - Example: **TF-IDF** (Term Frequency, Inverse Document frequency)

    - Weighted bag-of-words vector

    - One component per word in vocabulary
    - **Term frequency**: Number of times term appears in the document
    - **Document frequency**: Number of documents the term appears in

    $$
    \begin{array}{l}
    T F^{\prime}(d, t)=\log (1+T F(d, t)) \\\\
    I D F(t)=\log \frac{n_{d}-D F(t)}{D F(t)} \\\\
    T F I D F(d, t)=T F^{\prime}(d, t) I D F(t)
    \end{array}
    $$

- **Knowledge-based / semantic-based system**

  - Build semantic representation by extracting information from the question 
  - Construct structured query for semantic database

  - Not raw or indexed text corpus
  - Examples
    - WordNet

    - Wikipedia Infoboxes 
    - FreeBase

### Candidate Document Analysis

![截屏2020-09-18 14.50.32](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2014.50.32.png)

- Named entity tagging
  - Often including subclasses (towns, cities, provinces, ...)
- Sentence splitting, tagging, chunk parsing
- Identify multi-word terms and their variants
- Represent relation constraints of the text

### Answer Extraction 

![截屏2020-09-18 14.58.16](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2014.58.16.png)

- Input
  - Representations for candidate text segments and question 
  - Rank set of candidate sentences

  - Expected answer type(s)
- Find answer strings that match the answer type(s) based on documents 
  - Extractive: Answers are substrings in the documents

  - Generative: Answers are free text (NLG)
- Rank the candidate answers
  - E.g. overlap between answer and question 
- **Return result(s) with best overall score**

- Example

  ![截屏2020-09-18 15.00.35](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2015.00.35.png)

### Response Generation

![截屏2020-09-18 15.01.12](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2015.01.12.png)

- Rephrase text segment 
  - E.g. resolve anaphors

- Provide longer or shorter answer
  - Add some part of context into the answer

- If answer is too complex 
  - Truncate answer

  - Start dialog

## Neural Network Approach

- Neural models struggle with Information Retrieval 🤪
- **Excellent results on answer extraction** 😍 
  - Given: Question and Context (document, paragraph, nugget, etc.) 
  - Result: Answer as substring from context
    - Predict most likely start and end index as classification task
  - Combines:
    - Question Analysis
    - Retrieved Document Analysis
    - Answer Extraction
    - Response Generation

### Neural Answer Extraction

Encoder-decoder model

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2019.06.22.png" alt="截屏2020-09-18 19.06.22" style="zoom:80%;" />

Answer prediction

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2019.07.10.png" alt="截屏2020-09-18 19.07.10" style="zoom:80%;" />

- Softmax output $i$ is probability that answer starts at token $i$
- Mirrored setup for end probability
- 🔴 <span style="color:red">Problem: Relying on single vector for question encoding</span>
  - Long range dependencies 
  - Feedback at end of sequence 
  - Vanishing gradients

Solution: Use MORE information from the question

--> **Attention mechanism**

![截屏2020-09-18 19.09.26](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2019.09.26.png)

- Calculates weighted sum of question encodings
- Weight is based on similarity between question encoding and context encoding
- Different similarity metrics

