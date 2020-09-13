---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 160

# Basic metadata
title: "torch.nn"
date: 2020-09-10
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch"]
categories: ["Deep Learning"]
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
    pytorch:
        parent: getting-started
        weight: 6
---

## TL;DR

- torch.nn
  - `Module`: creates a callable which behaves like a function, but can also contain state(such as neural net layer weights). It knows what `Parameter` (s) it contains and can zero all their gradients, loop through them for weight updates, etc.
  - `Parameter`: a wrapper for a tensor that tells a `Module` that it has weights that need updating during backprop. Only tensors with the requires_grad attribute set are updated
  - `functional`: a module (usually imported into the `F` namespace by convention) which contains activation functions, loss functions, etc, as well as non-stateful versions of layers such as convolutional and linear layers.
- `torch.optim`: Contains optimizers such as `SGD`, which update the weights of `Parameter` during the backward step
- `Dataset`: An abstract interface of objects with a `__len__` and a `__getitem__`, including classes provided with Pytorch such as `TensorDataset`
- `DataLoader`: Takes any `Dataset` and creates an iterator which returns batches of data.

## Notebook

View in [nbviewer](https://nbviewer.jupyter.org/github/EckoTan0804/summay-pytorch/blob/master/pytorch-quick-start/06-what-is-torch_nn-exactly.ipynb)

## Reference

- [What is torch.nn *really*?](https://pytorch.org/tutorials/beginner/nn_tutorial.html#)