---
# Title, summary, and position in the list
linktitle: "12-Vision"
summary: ""
weight: 2120

# Basic metadata
title: "Language and Vision"
date: 2020-09-20
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
        weight: 13

---

## Motivation

Human interacts with environment multimodal 

- Modalities
  - Text
  - Audio 
  - Vision

- Other modalities can be used to disambiguate text 
- Jointly using different modalities

## Image description 

### Generation

- Generate description/caption of image 

  - Verbalize the most salient aspects of the image 

  - Typically one sentence

  - Example

    ![截屏2020-09-20 23.57.18](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2023.57.18.png)

- Joint use of

  - Computer vision
  - Natural language processing

#### 🔴 Challenges

- Cover any visual aspect of the image: 
  - Objects and their attributes

  - Features of the scene

  - Interaction of objects
- Reference to objects not in the image: 
  - E.g. *people waiting for a train*
- Background knowledge necessary 
  - E.g. *Picture of Mona Lisa*

#### Task

- Input: Image

- Generate representation

- Output: Text

- Related to Natural language generation 

  - Content selection

  - Organizing of content 
  - Surface realization

#### Generation from Visual Input

- Standard pipeline:
  1. **Computer vision**: Recognize
     - Scene
     - Objects

     - Spatial relationship 
     - Actions
  2. **Natural language generation**
     - Combine words/phrases from first step using
       - Templates 
       - N-grams 
       - Grammar rules

- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-21%2000.02.47.png" alt="截屏2020-09-21 00.02.47" style="zoom:80%;" />

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-21%2000.03.09.png" alt="截屏2020-09-21 00.03.09" style="zoom:80%;" />

- **End-to-End approaches** ([Show, Attend, Tell](https://arxiv.org/abs/1502.03044))

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-21%2000.05.48.png" alt="截屏2020-09-21 00.05.48" style="zoom:80%;" />

  - CNN Encoder of the image

  - LSTM-based Decoder generating the sentences

  - Attention mechanism to attend to different parts of the image

  - Examples

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-21%2000.06.17.png" alt="截屏2020-09-21 00.06.17" style="zoom:80%;" />

### Retrieval

- 💡 Idea: Use description of similar image

- Algorithm:

  - Extract visual feature

  - Retrieve most similar images using similarity function 
  - Re-rank images

  - Combine retrieved descriptions

- Example

  {{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-21%2000.08.07.png" title="Description retrieval" numbered="true" >}}

## Visual question answering

- Given: 

  - Image

  - Question related to the image

  - Example

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-21%2000.14.39.png" alt="截屏2020-09-21 00.14.39" style="zoom:67%;" />

- Output: Answer

- Most common model: Joint neural network

- 🔴 Challenges: Multi-step reasoning

- Steps

  1. Locate objects (*bike, window, street, basket and dogs*)
  2. Identify concepet (*sitting*)
  3. Rule out irrelavant objects

### Image model

![截屏2020-09-21 00.12.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-21%2000.12.48.png)

CNN:

- Often pretrained models used 
- Global features: Fixed size representation of the whole image 
- Local features: Representation of different regions of the image

### Text model

Read question word by word

![截屏2020-09-21 00.13.55](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-21%2000.13.55.png)

### Answer generation

- One word or free text

  - Input: Image features and text features

  - Output: Most probable word

- Models:

  - Fully connected NN 
  - Attention mechanism