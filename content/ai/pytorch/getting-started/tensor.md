---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 110

# Basic metadata
title: "Tensor"
date: 2020-09-07
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
        weight: 1
---

```python
import torch
```

- similar to Numpy's `ndarray`
- can also be used on a GPU to accelerate computing

## Operations

- similar to operations of Numpy's `ndarray`

  ```python
  x = torch.zeros(5, 3, dtype=torch.float)
  y = torch.randn_like(x, dtype=torch.float)
  
  x + y
  ```

- size: `size()`

  ```python
  x.size()
  ```

- in-place operation: post-fixed with an `_`

   ```python
  y.add_(x)
   ```

- Numpy-like indexing

  ```python
  x[:, 1]
  ```

- resize/reshape: `view()`

  ```python
  a = torch.ones(4, 4)
  b = a.view(-1, 8) # the size -1 is inferred from other dimensions
  # b should have size (2, 8)
  ```

- For one element tensor, use `.item` to get the value as a python number:

  ```python
  x = torch.rand(1)
  x.itemp
  ```



## PyTorch $\leftrightarrow$ Numpy

### Torch Tensor $\rightarrow$ Numpy Array

Use `numpy()`

```python
a = torch.ones(5) # torch tensor
b = a.numpy() # convert to numpy array
```

{{% alert warning %}} 

If both pytorch tensor and numpy array are on CPU, change one of them will change the another.

{{% /alert %}}

### Torch Tensor $\leftarrow$ Numpy Array

Use `from_numpy()`

```python
import numpy as np

a = np.ones(5) # numpy array
b = torch.from_numpy(a) # convert to torch tensor
```

{{% alert warning %}} 

Change the np array will change the torch tensor automatically

{{% /alert %}}

{{% alert note %}} 

Note: All the Tensors on the CPU **except a CharTensor** support converting to NumPy and back.

{{% /alert %}}

## CUDA Tensors

- Tensors can be moved onto any device using the `.to` method.
- Use `torch.device` objects to move tensors in and out of GPU

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device) # Create a tensor on GPU directly
    x = x.to(device) # Move a tensor to GPU use .to()
    z = x + y

    z.to("cpu", torch.double)
```

