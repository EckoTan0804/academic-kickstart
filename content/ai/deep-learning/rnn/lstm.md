---
# Title, summary, and position in the list
linktitle: "LSTM"
summary: ""
weight: 620

# Basic metadata
title: "Long Short-Term Memory (LSTM)"
date: 2020-08-21
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "RNN"]
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
        parent: rnn
        weight: 2
---

{{% alert note %}}
For detailed explanation and summary see: [LSTM Summary]({{< relref "../../natural-language-processing/RNN/lstm-summary.md" >}})
{{% /alert %}}

## Motivation

![截屏2020-08-21 12.10.50](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2012.10.50.png)

- Memory cell
- Inputs are “commited” into memory. Later inputs “erase” early inputs
- An additional memory “cell” for long term memory
- Also being read and write from the current step, but less affected like 𝐻

## LSTM Operations

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2012.16.54.png" alt="截屏2020-08-21 12.16.54" style="zoom:50%;" />

- Forget gate
- Input Gate
- Candidate Content
- Output Gate

### Forget

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2012.15.49.png" alt="截屏2020-08-21 12.15.49" style="zoom:50%;" />

- **Forget: remove information from cell $C$**
- What to forget depends on: 
  - the current input $X\_t$ 
  - the previous memory $H\_{t-1}$

- Forget gate: controls what should be forgotten
  $$
  F\_{t}=\operatorname{sigmoid}\left(W^{F X} X\_{t}+W^{F H} H\_{t-1}+b^{F}\right)
  $$

- Content to forget:
  $$
  C = F\_t * C\_{t-1}
  $$

  - $F\_{t,j}$ near 0: Forgetting the content stored in $C\_j$

### Write

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21 12.15.52.png" alt="截屏2020-08-21 12.15.52" style="zoom:50%;" />

- **Write: Adding new information for cell $C$**

- What to forget depends on: 

  - the current input $X\_t$ 
  - the previous memory $H\_{t-1}$

- Input gate: controls what should be add
  $$
  I\_{t}=\operatorname{sigmoid}\left(W^{I X} X\_{t}+W^{I H} H\_{t-1}+b^{I}\right)
  $$

- Content to write:
  $$
  \tilde{C}\_{t}=\tanh \left(W^{C X} X\_{t}+W^{C H} H\_{t-1}+b^{I}\right)
  $$

- Write content:
  $$
  C=C\_{t-1}+I\_{t} * \tilde{C}\_{t}
  $$

### Output

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2012.15.54.png" alt="截屏2020-08-21 12.15.54" style="zoom:50%;" />

- **Output: Reading information from cell 𝐶 (to store in the current state $H$)**

- How much to write depends on:

  - the current input $X\_t$ 
  - the previous memory $H\_{t-1}$

- Forget gate: controls what should be output
  $$
  O\_{t}=\operatorname{sigmoid}\left(W^{OX} X\_{t}+W^{OH} H\_{t-1}+b^{O}\right)
  $$

- New state
  $$
  H\_t = O\_t * \operatorname{tanh}(C\_t)
  $$
  

## LSTM Gradients

![截屏2020-08-21 12.16.32](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2012.16.32.png)

## Truncated Backpropagation

What happens if the sequence is really long (E.g. Character sequences, DNA sequences, video frame sequences ...)?
$\to$ Back-propagation through time becomes *exorbitant* at large $T$

Solution: **Truncated Backpropagation**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2012.49.17.png" alt="截屏2020-08-21 12.49.17" style="zoom: 67%;" />

- Divide the sequences into segments and truncate between segments 
- However the memory is kept to remain some information about the past (rather than resetting)

## TDNN vs. LSTM

|                                          | TDNN                                                         | LSTM                                                         |
| ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                          | <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2012.37.38.png" alt="截屏2020-08-21 12.37.38" style="zoom:80%;" /> | <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2012.37.53.png" alt="截屏2020-08-21 12.37.53" style="zoom:150%;" /> |
|                                          | For $t = 1,\dots, T$:<br />$H\_{t}=\mathrm{f}\left(\mathrm{W}\_{1} \mathrm{I}\_{\mathrm{t}-\mathrm{D}}+\mathrm{W}\_{2} \mathrm{I}\_{\mathrm{t}-\mathrm{D}+1}+\mathrm{W}\_{3} \mathrm{I}\_{\mathrm{t}-\mathrm{D}+2}+\cdots+W\_{D} I_{t}\right)$ | For $t = 1,\dots, T$:<br />$H\_{t}=f\left(W^{I} I\_{t}+W^{H} H\_{t-1}+b\right)$ |
| Weights are shared over time?            | Yes                                                          | Yes                                                          |
| Handle variable length sequences         | <li>Increasing long-range dependency learning by increasing $D$ <br /><li>Increasing $D$ also increases number of parameters | Can flexibly adapt to variable length sequences without changing structures |
| Gradient vanishing or exploding problem? | No                                                           | Yes                                                          |
| Parallelism?                             | Can be parallelized into 𝑂(1)<br />(*Assuming Matrix multiplication cost is O(1) thanks to GPUs...*) | Sequential computation<br />(because $𝐻\_𝑇$ cannot be computed before $𝐻\_{𝑇−1}$) |

