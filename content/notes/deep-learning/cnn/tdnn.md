---
# Title, summary, and position in the list
linktitle: "TDNN"
summary: ""
weight: 410

# Basic metadata
title: "Time-Delay Neural Network (TDNN)"
date: 2020-08-16
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Optimization", "CNN"]
categories: ["Deep Learning"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?

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
    deep-learning:
        parent: cnn
        weight: 1

---

## Motivation

Ensure **shift-invariance**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-19%2011.45.55.png" alt="截屏2020-08-19 11.45.55" style="zoom:80%;" />

## Overview

- Multilayer Neural Network: **Nonlinear** Classifier

- Consider **Context** (Receptive Field)
- **Shift-Invariant** Learning
  - All Units Learn to Detect Patterns *Independent* of Location in Time
  - No Pre-segmentation or Pre-alignment Necessary
  - Approach: **Weight Sharing**
- **Time-Delay Arrangement** 
  - Networks can represent temporal structure of speech
- **Translation-Invariant Learning** 
  - Hidden units of the network learn features independent of precise location in time

## Structure

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-19%2011.50.41.png" alt="截屏2020-08-19 11.50.41" style="zoom: 50%;" />

- Input: spectrum of a speech
  - $x$-axis: time
  - $y$-axis: frequency

### How TDNN works?

#### Input layer $\to$ Hidden layer

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-19%2011.53.08.png" alt="截屏2020-08-19 11.53.08" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-19%2011.53.12.png" alt="截屏2020-08-19 11.53.12" style="zoom:50%;" />

####  Hidden layer 1 $\to$  Hidden layer 2

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-19 11.54.03.png" alt="截屏2020-08-19 11.54.03" style="zoom:50%;" />

- As this input flows by, we have these hidden units generated activations over time as activation patterns.
- Then we can take a contextual window of activation patterns over time and feed them into neurons in the second hidden layer 

#### Hidden layer $\to$ Output layer

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-19%2011.54.37.png" alt="截屏2020-08-19 11.54.37" style="zoom:50%;" />

- We assemble all the evidence from activations over time and integrate them into one joint output

## Shift-Invariance Training

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-19%2012.00.24.png" alt="截屏2020-08-19 12.00.24" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/20131123104203421.png" alt="img" style="zoom:75%;" />

Connections with the same color share the same weight.

## Demo

[TDNN / Convolutional Nets - Demo](https://lecture-demo.ira.uka.de/)

## TDNN’s→Convolutional Nets

In Vision the same problem:

- Local Contexts – Global Integration – Shared Weights

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-19%2012.05.36.png" alt="截屏2020-08-19 12.05.36" style="zoom:67%;" />

TDNN is equivalent to 1-dimensional CNN



## Reference

- [Neural Networks for Acoustic Modelling 3: Context-dependent DNNs and TDNNs](http://www.inf.ed.ac.uk/teaching/courses/asr/2018-19/asr09-dnn.pdf)

- [语音识别——TDNN时延神经网络](https://blog.csdn.net/qq_14962179/article/details/87926351)

