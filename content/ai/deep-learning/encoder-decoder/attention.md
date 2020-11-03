---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 620

# Basic metadata
title: "👍 Attention"
date: 2020-08-16
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Seq2Seq"]
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
        parent: encoder-decoder
        weight: 2

---

## Core Idea

The main assumption in sequence modelling networks such as RNNs, LSTMs and GRUs is that **the current state holds information for the whole of input** seen so far. Hence the final state of a RNN after reading the whole input sequence should contain complete information about that sequence. But this seems to be too strong a condition and too much to ask.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*1JcHGUU7rFgtXC_mydUA_Q.jpeg" alt="Image for post" style="zoom: 33%;" />



Attention mechanism relax this assumption and proposes that **we should look at the hidden states corresponding to the whole input sequence in order to make any prediction.**



## Details

The architecture of attention mechanism:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*e5665dfyxLDgZzKmrZ8Y0Q.png" alt="Image for post" style="zoom: 40%;" />

The network is shown in a state: 

- the encoder (lower part of the figure) has computed the hidden states $h\_j$ corresponding to each input $X\_j$ 
- the decoder (top part of the figure) has run for $t-1$ steps and is now going to produce output for time step $t$.

The whole process can be divided into four steps:

1. [Encoding](#encoding)
2. [Computing Attention Weights/Alignment](#computing-attention-weightsalignment)
3. [Creating context vector](#creating-context-vector)
4. [Decoding/Translation](#decodingtranslation)

### Encoding

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*_hL6bQGbYGSJ4E-PgF4UfA.png" alt="Image for post" style="zoom:33%;" />



- $(X\_1, X\_2, \dots, X\_T)$: Input sequence

  - $T$: Length of sequence

- $(\overrightarrow{h}\_{1}, \overrightarrow{h}\_{2}, \dots, \overrightarrow{h}\_{T})$: Hidden state of the forward RNN

- $(\overleftarrow{h}\_{1}, \overleftarrow{h}\_{2}, \ldots \overleftarrow{h}\_{T})$: Hidden state of the backward RNN

- The hidden state for the $j$-th input $h\_j$ is the *concatenation* of $j$-th hidden states of forward and backward RNNs.

  $$
  h\_{j}=\left[\overrightarrow{h}\_{j} ; \overleftarrow{h}\_{j}\right], \quad \forall j \in[1, T]
  $$

### Computing Attention Weights/Alignment

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*jiJmd9ako4eBBkEf0igTHA.png" alt="Image for post" style="zoom:33%;" />



At each time step $t$ of the decoder, the amount of attention to be paid to the hidden encoder unit $h\_j$ is denoted by $\alpha_{tj}$ and calculated as a function of both $h\_j$ and previous hidden state of decoder $s\_{t-1}$:
$$
\begin{array}{l}
e\_{t j}=\boldsymbol{a}\left(h\_{j}, s\_{t-1}\right), \forall j \in[1, T] \\\\ \\\\
\alpha_{t j}=\frac{\displaystyle \exp \left(e\_{t j}\right)}{\displaystyle \sum_{k=1}^{T} \exp \left(e\_{t k}\right)}
\end{array}
$$

- $\boldsymbol{a}(\cdot)$: parametrized as a feedforward neural network that runs for all $j$ at the decoding time step $t$
- $\alpha\_{tj} \in [0, 1]$
- $\displaystyle \sum\_j \alpha\_{tj} = 1$
- $\alpha\_{tj}$ can be visualized as the attention paid by decoder at time step $t$ to the hidden ecncoder unit $h\_j$

### Computing Context Vector

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*Y78e7OLg9A4LAg3_4bRUGA.png" alt="Image for post" style="zoom:33%;" />



Now we compute the context vector. The context vector is simply a linear combination of the hidden weights $h\_j$ weighted by the attention values $\alpha_{tj}$ that we've computed in the precdeing step:
$$
c\_t = \sum\_{j=1}^T \alpha\_{tj}h\_j
$$
From the equation we can see that $\alpha_{tj}$ determines how much $h\_j$ affects the context $c\_t$. The higher the value, the higher the impact of $h\_j$ on the context for time $t$.

### Decoding/Translation

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*uIiUT02LY8aa5Qj4rUZ8Uw.png" alt="Image for post" style="zoom:33%;" />

Compute the new hidden state $s\_t$ using

- the context vector $c\_t$
- the previous hidden state of the decoder $s\_{t-1}$
- the previous output $y\_{t-1}$

$$
s\_{t}=f\left(s\_{t-1}, y\_{t-1}, c\_{t}\right)
$$

The output at time step $t$ is
$$
p\left(y\_{t} \mid y\_{1}, y\_{2}, \ldots y\_{t-1}, x\right)=g\left(y\_{t-1}, s\_{t}, c\_{i}\right)
$$
{{% alert note %}}
In the [paper](https://arxiv.org/pdf/1409.0473.pdf), authors have used a GRU cell for $f$ and a similar function for $g$.
{{% /alert %}}

## Reference

- [Understanding Attention Mechanism](https://medium.com/@shashank7.iitd/understanding-attention-mechanism-35ff53fc328e)
- [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)
- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) 👍



