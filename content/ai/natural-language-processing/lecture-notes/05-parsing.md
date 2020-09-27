---
# Title, summary, and position in the list
linktitle: "05-Parsing"
summary: ""
weight: 2060

# Basic metadata
title: "Parsing"
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
        weight: 6

---

![截屏2020-09-16 23.41.41](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2023.41.41.png)

## TL;DR

- Representing and Analyze Sentence Structure

- Phrase structure grammar
  - Context free grammar
  - Problems:
    - Ambiguities : PP Attachment

- Traditional Approaches 
- Stochastically Parsing
  - Probabilistic Context Free Grammar 
  - CYK Algorithm

  - Transition-based parsing

## Grammaticality

Common approach in statistical natural language processing: **n-gram Language Model**

E.g., tri-gram
$$
\begin{array}{l}
P\left(w_{1}, \ldots, w_{n}\right) \\
=P\left(w_{1}\right) * P\left(w_{2} \mid w_{1}\right) * P\left(w_{3} \mid w_{1} w_{2}\right) \ldots \\
\approx P\left(w_{n} \mid w_{n-2} w_{n-1}\right)
\end{array}
$$
<span style="color:red">Problems of Language Models</span>

- <span style="color:red">Generalization: even with very long context there are sentence you cannot model with a n-gram language model</span>
- <span style="color:red">Overall sentence structure</span>

How can we model what a grammatically correct sentence is?

- Need arbitrary context
- Use grammar describing generation of the sentence

## Phrase structure grammar 

**Describe sentence structure by grammar** (Constituency relation)

Phrase structure organizes words into nested constituents (can represent the grammar with [CFG](#context-free-grammar) rules)

Units in the grammar: **Constituency**

- Can be moved around
  - *I saw you <u>today</u>*
  - *<u>Today</u>, I saw you*
- expand/contract
  - *I saw <u>the boy</u>*
  - *I saw <u>him</u>*
  - *I saw <u>the old boy*</u>

> [Wiki](https://en.wikipedia.org/wiki/Constituent_(linguistics))
>
> In syntactic analysis, a **constituent** is a word or a group of words that function **as a single unit** within a hierarchical structure. The constituent structure of sentences is identified using tests for constituents.
>
> A phrase is a sequence of one or more words (in some theories two or more) built around a head lexical item and working as a unit within a sentence. A word sequence is shown to be a phrase/constituent if it exhibits one or more of the behaviors discussed below. 

### Phrase structure rules

- Describe syntax of language
- Example
  - `s` -->`NP` `VP` (Sentence consists of a noun phrase and a verb phrase) 
  - `NP` --> `Det` `N` (A noun phrase consists of a determiner and a noun)
- **Only looking at the syntax** 
- **No semantics**

> [Wiki](https://en.wikipedia.org/wiki/Phrase_structure_grammar):
>
> In linguistics, phrase structure grammars are all those grammars that are based on the **constituency relation**, as opposed to the dependency relation associated with **dependency grammars**; hence, phrase structure grammars are also known as **constituency grammars**
>
> The fundamental trait that these frameworks all share is that they view sentence structure in terms of the constituency relation. 
>
> Example: Constituency relation Vs. Dependency relation
>
> ![Constituency and dependency relations](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Thistreeisillustratingtherelation%28PSG%29.png)

### Context Free Grammar

**Constituency = phrase structure grammar = context-free grammars (CFGs)**

- Introduced by Chomsky

- 4-tuple: 
  $$
  G = (V, \Sigma, R, S)
  $$

  - $V$: finite set of non-terminals
    - variables describing the phrases (NP, VP, ...)
  - $\Sigma$: finite set of terminals
    - content of the sentence

    - all words in the grammar
  - $R$: finite relation $V$ to $(V \cup \Sigma)^{\*}$
    - Rules defining how non-terminals can be replaced
    - E.g.: `s` -->`NP` `VP`
  - $S$: start symbol

*Example*

![截屏2020-09-27 16.05.45](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-27%2016.05.45.png)

## Dependency Structure

- Different approach to describe sentence structure

- Identify semantic relations!

- Idea:
  - Which words depend on which words 
  - Which word modifies which word
  
- Example:

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-27%2016.20.20.png" alt="截屏2020-09-27 16.20.20" style="zoom: 67%;" />

> [Wiki](https://en.wikipedia.org/wiki/Dependency_grammar)
>
> The (finite) verb is taken to be the structural center of clause structure. All other syntactic units (words) are either directly or indirectly connected to the verb in terms of the directed links, which are called dependencies.
>
> A dependency structure is determined by the relation between a word (a head) and its dependents. Dependency structures are flatter than phrase structures in part because they lack a finite verb phrase constituent, and they are thus well suited for the analysis of languages with free word order, such as Czech or Warlpiri.

## Difficulties

**<span style="color:red">Ambiguities!!!</span>**

E.g.: Prepositional phrase attachment ambiguity

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2016.34.55.png" alt="截屏2020-09-16 16.34.55" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2016.35.42.png" alt="截屏2020-09-16 16.35.42" style="zoom:67%;" />

 ## Parsing

**Automatically generate parse tree for sentence**

- Given:

  - Grammar

  - Sentence

- Find: hidden structure

- Idea: Search for different parses

Applications

- Question – Answering

- Named Entity extraction 
- Sentiment analysis

- Sentence Compression

### Traditional approaches 

**Hand-defined rules**: restrict rules by hand to have at best only one possible parse tree

🔴 Problems

- Many parses for the same sentence
- Coverage Problem (Many sentences could not be parsed)
- Time and cost intensive

### Statistical parsing

Use **machine learning techniques** to distinguish probable and less probable trees

- Automatically learn rules from training data 
  - Hand-annotated text with parse trees
- still many parse trees for one sentence 🤪
  - But weights define most probable
- Tasks
  - **Training**: learn possible rules and their probabilities
  - **Search**: find most probable parse tree for sentence

#### Annotated Data

**Treebank**:

- human annotated sentence with structure 
  - Words
  - POS Tags

  - Phrase structure

- 👍 Advantages: 

  - Reusable

  - High coverage 
  - Evaluation

- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2017.42.49.png" alt="截屏2020-09-16 17.42.49" style="zoom:80%;" />

#### **Probabilistic Context Free Grammar**

- Extension to Context Free Grammar

- Formel definition: 5 tuple
  $$
  G = (V, \Sigma, R, S, P)
  $$

  - $V, \Sigma, R, S$: same as Context Free Grammar
  - $P$: set of Probabilities on production rules
    - E.g.: `s` -->`NP` `VP` 0.5

- Properties

  - Probability of derivation is product over all rules
    $$
    P(D)=\prod_{r \in D} P(r)
    $$

  - Sum over all probabilities of rules replacing one non-terminal is one
    $$
    \sum_{A} P(S \rightarrow A)=1
    $$

  - Sum over all derivations is one
    $$
    \sum_{D \in S} P(D)=1
    $$

#### Training

- Input: Annotated training data (E.g.: Treebank)
- Training
  - **Rule extraction**: Extract possible rules from the trees of the training data
  - **Probability estimation**
    - Assign probabilities of the rules 
    - Maximum-likelihood estimation

- Example:

  ![截屏2020-09-16 18.35.34](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2018.35.34.png)

#### Search

- Find possible parses of the sentence 
- Statistical approach: Find all/many possible parse trees 
- Return most probable one

- Strategies:

  - Top-Down

  - Bottom up

    - **Shift reduce** algorithm

      - **Shift**: advances in the input stream by one symbol. That shifted symbol becomes a new single-node parse tree.
      - **Reduce**: applies a completed grammar rule to some of the recent parse trees, joining them together as one tree with a new root symbol.

    - Example

      {{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/56cee123595482cf3edaef089cb9a6a7.jpg" title="Shift reducea algorithm example" numbered="true" >}}

  - Dynamic Programming

### CYK Parsing

- Avoid repeat work

- Use Dynamic Programming

  - Transform grammar in Chomsky normal form 
  - Store best trees for subphrases
  - Combine tree from best trees of subphrases

- All rules must have the following form

  - A --> BC
    - A, B, C non-terminals
    - B, C not the start symbol
  - A --> a
    - A non-terminal
    - a terminal
  - S --> $\epsilon$
    - Create empty string if it is in the grammar

- Every context-free grammar can be transferred into one having Chomsky normal form

  - Binarization

    - Only rules with two non-terminals

    - Idea:

      - Introduce additional non-terminal

      - Replace one rules with three non-terminals by two rules with two non- terminals each

    - Example

      <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2021.32.16.png" alt="截屏2020-09-16 21.32.16" style="zoom:67%;" />

      

  - Remove unaries

    - Remove intermediate rules

- Problems

  - Very strong indepedence assumption
  - Label is bottleneck
  
- [Example](https://en.wikipedia.org/wiki/CYK_algorithm)

  - Grammar

    ![截屏2020-09-27 23.40.08](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-27%2023.40.08.png)

  - Analyse the sentence "*she eats a fish with a fork*" with the CYK algorithm:

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/440px-CYK_algorithm_animation_showing_every_step_of_a_sentence_parsing.gif" alt="img" style="zoom:80%;" />

    result:

    ![截屏2020-09-27 23.41.25](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-27%2023.41.25.png)

### Transition-based Dependency Parsing

Model Dependency structure

![截屏2020-09-16 22.08.00](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.08.00.png)

Predict transition sequence: Transition between configuration

#### **Arc-standard System**

- Configuration

  - Stack
  - Buffer
  - Set of Dependency Arcs
  - Initial configuration: [Root], $w_1,\dots, w_n$, {}
    - All words are in the buffer
    - The stack is empty
    - The dependency graph is empty
  - Terminal configuration
    - The buffer is empty
    - The stack contains a single word

- Example

  ![截屏2020-09-16 22.00.17](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.00.17.png)

- Transistions

  - Left-arc
    $$
    ([\sigma|i| j], B, A) \Rightarrow([\sigma \mid j], B, A \cup\{j, I, i\})
    $$

    - Add dependency between top and second top element of the stack with label l to the arcs
    - Remove second top element from the stack

  - Right-arc
    $$
    ([\sigma|i| j], B, A) \Rightarrow([\sigma \mid i], B, A \cup\{i, I, j\})
    $$

    - Add dependency between second top and top element of the stack with label l to the arcs
    - Remove top element from the stack

  - Shift: Move first elemnt of the buffer to the stack

- Example

  - Initial configuration

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.14.32.png" alt="截屏2020-09-16 22.14.32" />

  - Shift

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.14.35.png" alt="截屏2020-09-16 22.14.35" />

  - Shift

  ![截屏2020-09-16 22.14.38](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.14.38.png)

  - Left arc

  ![截屏2020-09-16 22.14.42](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.14.42.png)

  - Shift

  ![截屏2020-09-16 22.14.49](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.14.49-20200916222538371.png)

  - Shift

  ![截屏2020-09-16 22.14.52](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.14.52.png)

  

  - Left arc

  ![截屏2020-09-16 22.15.08](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.15.08-20200916221721930.png)

  - Shift

  ![截屏2020-09-16 22.15.11](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.15.11-20200916221740541.png)

  - Right arc

  ![截屏2020-09-16 22.15.16](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.15.16.png)

  - Right arc

    ![截屏2020-09-16 22.25.12](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2022.25.12.png)

##### Problems

- Sparsity
- Incompleteness
- Expensive computation



#### Neural Network-based prediction

- Feed forward neural network to predict operation

- Inpupt

  - Set of words $S^w$, pos-tags $S^t$ adn labels $S^l$
  - Fixed number
  - Map to continuous space

- Output

  - Operation
  - $2N_l + 1$

- Example structure

  ![截屏2020-09-16 23.33.02](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-16%2023.33.02.png)

## Evaluation

- Label precision/recall

- Describe tree as set of triple (Label, start, end)

- Calculate precision/recall/f-score of reference and hypothesis



## Reference

- Shift reduce algorithm
  - [Shift reduce Parsing](https://www.bookstack.cn/read/nlp-py-2e-zh/spilt.4.8.md)
  - [Wiki](https://en.wikipedia.org/wiki/Shift-reduce_parser)
- [Transition-based dependency parsing](https://cl.lingfil.uu.se/~sara/kurser/5LN455-2014/lectures/5LN455-F8.pdf)

- [[CS224n笔记] L5 Dependency Parsing](https://zhuanlan.zhihu.com/p/110532288)
- [CS224n, Linguistic Structure: Dependency Parsing](http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture05-dep-parsing.pdf)