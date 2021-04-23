---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 402

# Basic metadata
title: "Transformer"
date: 2021-04-11
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Transformer", "Attention"]
categories: ["Computer Vision"]
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
    computer-vision:
        parent: cv-transformer
        weight: 2
---

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/transformer-stey_by_step%20%283%29.png" caption="Transformer architecture" numbered="true" >}}

## Reference

- [Transformer: A Novel Neural Network Architecture for Language Understanding](http://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) - An introduction of Transformer from Google AI Blog

- Tutorials
  - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Detailed explanation with tons of illustrations 👍🔥

  - [Illustrated Guide to Transformers Neural Network: A step by step explanation](https://www.youtube.com/watch?v=4Bdc55j80l8) - Step-by-step video explanation 📹👍🔥

    {{< youtube 4Bdc55j80l8>}}

  - [LSTM is dead. Long Live Transformers!](https://www.youtube.com/watch?v=S27pHKBEp30) - Briefly explanation of Transformer 

  - [Attention Is All You Need](https://www.youtube.com/watch?v=iDulhoQ2pro) - Video explanation about the paper "Attention is All You Need"

- Visualization

  - [Getting meaning from text: self-attention step-by-step video](https://peltarion.com/blog/data-science/self-attention-video) ([video](https://www.youtube.com/watch?v=-9vVhYEXeyQ&t=4s))

- Implementation

  - [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - A guide annotation the paper with PyTorch implementation 👍🔥
  - [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) - PyTorch implementation