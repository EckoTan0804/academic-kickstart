---
# Title, summary, and position in the list
linktitle: "08-NLU"
summary: ""
weight: 2090

# Basic metadata
title: "Natural/Spoken Language Understanding"
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
        weight: 9

---

![截屏2020-09-18 22.29.24](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2022.29.24.png)



## Definition

**Natural language understanding**

- Representing the semantics of natural language 
- Possible view: Translation from natural language to representation of meaning

<span style="color:red">Difficulties</span>

- Ambiguities 
  - Lexical

  - Syntax
  - Referential 
- Vagueness
  - E.g., *"I had a late lunch."*
- Dimensons
  - Depth: Shallow vs Deep 
  - Domain: Narrow vs Open

## Examples

### Siri (2011)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-19%2012.02.45.png" alt="截屏2020-09-19 12.02.45" style="zoom:50%;" />

 ## Dialog Modeling

**Dialog system / Conversational agent**

- Computer system that converse with a human 
- Coherent structure

- Different modalities:
  - Text, speech, graphics, haptics, gestures

### Components

![截屏2020-09-18 22.58.28](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2022.58.28.png)

#### Input recognition

- Different modalities

  - **Automatic speech recognition (ASR)**
  - Gesture recognition

- Transformation

  Input modality (e.g. speech) --> text

- May introduce first errors
- High influence on the performance of an dialog system

#### Natural language understanding (NLU)

-  Semantic interpretation of written text

- Transformation from natural language to semantic representation 
- Representations:
  - Deep vs Shallow

  - Domain-dependent vs. domain independent

#### Dialog manager (DM)

- **Manage flow of conversation** 
- Input: Semantic representation of the input 
- Output: Semantic representation of the output 
- Utilize additional knowledge
  - User information

  - Dialog History 
  - Task-specific information

#### Natural language generation (NLG)

**Generate natural language from semantic representation**

- Input: Semantic output representation of the dialog manager
- Output: Natural language text for the user

#### Output rendering

**Generate correct output**

- e.g. **Text-to-Speech (TTS)** for Spoken Dialog Systems

 ## Natural Language understanding 

### Approaches

![截屏2020-09-18 23.09.46](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2023.09.46.png)

#### **Output representation**

- **Relation instances**
  - (Larry Page, founder, Google) 
- **Logical forms**
  - Love(Mary, John) 
- **Scalar**
  - Positive/Negative 0.9 
- **Vector**
  - Hidden representation/ Word embeddings

#### Algorithms

- Rule-based / Template
- Machine learning
  - Conditional random fields
  - Support Vector Machine
  - Neural Networks / Deep learning

 ### Semantic Parsing

- **Parse natural language sentence into semantic representation** 
- Machine learning approaches most successful :clap:

- Most common approach:
  - Shallow Semantic Parsing / Semantic Role Labeling
- Most important resources: 
  - [PropBank](#propbank)
  - [FrameNet](#framenet)

#### PropBank

- **Proposition Bank (PropBank)**

- Labels for all sentence in the English Penn TreeBank 

- Defines semantic based on the verbs of the sentence

- **Verbs**: Define different senses of the verbs

- **Sense**: Number of Arguments important to this sense (Often only numbers)

  - Arg0: Proto-Agent

  - Arg1: Proto-Patient

  - Arg2: mostly benefactive, instrument, attribute, or end state 
  - Arg3: start point, benefactive, instrument, or attribute

##### Example: "agree"

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-18%2023.16.28.png" alt="截屏2020-09-18 23.16.28" style="zoom:67%;" />

##### Example: "fall"

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-19%2010.52.24.png" alt="截屏2020-09-19 10.52.24" style="zoom: 40%;" />

##### PropBank ArgM

- `TMP` : when? *yesterday evening, now*
- `LOC` : where? *at the museum, in San Francisco*
- `DIR` : where to/from? *down, to Bangkok*
- `MNR` : how? *clearly, with much enthusiasm*
- `PRP/CAU` : why? *because ... , in response to the ruling* 
- `REC` : *themselves, each other*
- `ADV` : *miscellaneous*
- `PRD` : secondary predication ...*ate the meat raw*

##### 🔴 Problem

Different words, Predicate expressed by noun

Example

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-19%2010.59.11.png" alt="截屏2020-09-19 10.59.11" style="zoom: 40%;" />

{{% alert note %}} 

More see: [SRL数据集(1): Proposition Bank 数据集介绍](https://zhuanlan.zhihu.com/p/37254041)

{{% /alert %}}

#### FrameNet

- **Roles based on Frames**
- **Frame**: holistic background knowledge that unites these words
- **Frame-elements**: Frame-specific semantic roles

##### Example 1

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-19%2011.08.49.png" alt="截屏2020-09-19 11.08.49" style="zoom: 40%;" />

##### Example 2

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-19%2011.19.24.png" alt="截屏2020-09-19 11.19.24" style="zoom: 40%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-19%2011.18.40.png" alt="截屏2020-09-19 11.18.40" style="zoom: 33%;" />

### Semantic Role labeling

- Task: Automatically finding semantic roles for each argument of each predicate

- Approach: Maching Learning

- High level algorithm:

  1. Parse sentence (Syntax tree)

  2. Find Predicates

  3. For every node in tree: Decide semantic role

## Spoken Language understanding

**Natural language processing for spoken input**

 ### Difficulties 

- **Less grammatically speech**

  - Partial Sentences

  - Disfluencies (Self correction, hesitations, repetitions)

- **Robust to noise** 

  - ASR errors

  - Techniques:
    - Confidence
    - Multiple hypothesis

- **No Structure information**

  - Punctuation

  - Text segmentation

### Approach

- Transform text into task-specific semantic representation of the user’s intention
- Subtasks
  - [Domain detection](#domain-detection) 
  - [Intention determination](#intention-determination) 
  - [Slot filling](#slot-filling)

#### Domain Detection

- Motivated by **Call Centers**
  - Many agents with specialization on one topic (*Billing inquiries, technical support requests, sales inquiries, etc.*)
- First techniques: Menus to find appropriate agent
- Automatic task:
  - Given the utterance find the correct agent Utterance classification task
  - **Utterance classification task**
- Input: Utterance
- Output: Topic

#### **Intention determination**

- Domain-dependent utterance classes 
  - e.g. Find_Flight
- Task: Assign class to Utterance
- Use similar technique

#### Slot filling

**Sequence labeling task**: Assign semantic class label to every word and history

- History: previous words and labels

Example:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-19%2011.56.02.png" alt="截屏2020-09-19 11.56.02" style="zoom: 50%;" />

Success of deep learning in other approaches:

- RNN-based approach
- Find most probable label given word and history