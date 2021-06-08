---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 150

# Basic metadata
title: "Learn PyTorch with Example"
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
        weight: 5
---

## TL;DR

- PyTorch provides two main features:
  - An n-dimensional Tensor, similar to numpy but can run on GPUs
  - Automatic differentiation for building and training neural networks

- Typical procedure of neural network training with PyTorch

  1. Define network structure

     - Use `torch.nn.Sequential`, e.g.: 

       ```python
       model = torch.nn.Sequential(
           torch.nn.Linear(D_in, H),
           torch.nn.ReLU(),
           torch.nn.Linear(H, D_out),
       )
       ```

       or

     - Define own Modules by

       - subclassing `nn.Module`
       - defining a `forward` function which receives input Tensors

       ```python
       import torch
       
       class TwoLayerNet(torch.nn.Module):
           
           def __init__(self, D_in, H, D_out):
               """
               In the constructor we instantiate two nn.Linear modules and assign them as
               member variables.
               """
               super(TwoLayerNet, self).__init__()
               self.linear1 = torch.nn.Linear(D_in, H)
               self.linear2 = torch.nn.Linear(H, D_out)
       
           def forward(self, x):
               """
               In the forward function we accept a Tensor of input data and we must 
               return a Tensor of output data. 
               We can use Modules defined in the constructor as well as arbitrary 
               operators on Tensors.
               """
               h_relu = self.linear1(x).clamp(min=0)
               y_pred = self.linear2(h_relu)
               return y_pred
       ```

  2. Define loss function and optimizer (and learning rate)

     - Loss function: implemented in [`torch.nn`](https://pytorch.org/docs/stable/nn.html#loss-functions)

       - E.g.: Mean Square Loss

         ```python
         loss_fn = torch.nn.MSELoss(reduction='sum')
         ```

     - Optimizer (see: [`torch.optim`](https://pytorch.org/docs/stable/optim.html)) and learning rate

       - E.g.: Adam

         ```python
         learning_rate = 1e-4
         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
         ```

   3.  Iterate training dataset multiple times. In each iteration

        	1. Forward pass
        	2. Compute loss
        	3. Zero all of the parameters' gradients 
        	4. Backward pass
        	5. Update parameters

       ```python
       for t in range(500):
           # 3.1 Forward pass
           y_pred = model(x)
       
           # 3.2 Compute and print loss
           loss = loss_fn(y_pred, y)
           if t % 100 == 99:
               print(t, loss.item())
       
       		# 3.3 Zero gradients
           optimizer.zero_grad()
       
           # 3.4 Backward pass
           loss.backward()
       
           # 3.5 Update parameters
           optimizer.step()
       ```

### Diagramm Summary

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/pytorch_train.png" title="Training in PyTorch overview" numbered="true" >}}

## From `numpy` to `pytorch`

View in [nbviewer](https://nbviewer.jupyter.org/github/EckoTan0804/summay-pytorch/blob/master/pytorch-quick-start/05-learn-pytorch-with-examples.ipynb)

## Reference

- [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#)