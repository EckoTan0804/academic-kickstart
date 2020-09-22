---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 170

# Basic metadata
title: "Visualization with TensorBoard"
date: 2020-09-22
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
        weight: 7
---

## TL;DR

0. Define network structure, loss function, optimizer

1. Set up TensorBoard

   1. Import `tensorboard` from `torch.utils`

   2. Define `SummaryWriter`

      

   ```python
   from torch.utils.tensorboard import SummaryWriter
   
   # let's say we'll log for "fashion_mnist_experiment_1"
   NAME = "fashion_mnist_experiment_1"
   
   # default `log_dir` is "runs"
   writer = SummaryWriter(f'runs/{NAME}')
   ```

2. Launch TensorBoard

   1. If we launch in Jupyter Notebook/Lab or Google Colab, we need to load the TensorBoard notebook extension first

      ```python
      %load_ext tensorboard
      ```

   2. Launch TensorBoard

      ```python
      %tensorboard --logdir=runs
      ```

3. Inspect the model 

   - using `add_graph()` 

   - remember to call `writer.close()`

   

   ```python
   # assume that net is our neural network
   # and images are a batch of training images
   
   writer.add_graph(net, images)
   writer.close()
   ```

4. Track model  training with `add_scalar()`

5. Access trained models

   - Precision-Recall curve: `add_pr_curve()`

   

## Notebook

[04-visualize-with-tensorboard.ipynb](https://github.com/EckoTan0804/summay-pytorch/blob/master/pytorch-quick-start/04-visualize-with-tensorboard.ipynb)

## Reference

- [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#)