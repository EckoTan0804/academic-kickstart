---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 180

# Basic metadata
title: "🤔 PyTorch Understanding"
date: 2020-09-25
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
        weight: 8
---

## `squeeze` and `unsqueeze`

Simply put, `unsqueeze()` "adds" superficial `1` dimension to tensor (at specified dimension), while `squeeze` removes all superficial `1` dimensions from tensor.

### `squeeze`

```python
torch.squeeze(input, dim=None, out=None) → Tensor
```

- Documentation: [`torch.squeeze`](https://pytorch.org/docs/stable/generated/torch.squeeze.html)
- Returns a tensor with all the dimensions of `input` of size 1 removed.
  - For example, if input is of shape: $(A \times 1 \times B \times C \times 1 \times D)$ then the out tensor will be of shape: $(A \times B \times C \times D)$.
  - When `dim` is given, a squeeze operation is done only in the given dimension. 
    - If input is of shape: $(A \times 1 \times B)$, `squeeze(input, 0)` leaves the tensor unchanged
    - But `squeeze(input, 1)` will squeeze the tensor to the shape $(A \times B)$ .

- Example

  ```python
  >>> x = torch.zeros(2, 1, 2, 1, 2)
  >>> x.size() # alternative: x.shape
  torch.Size([2, 1, 2, 1, 2])
  >>> y = torch.squeeze(x)
  >>> y.size()
  torch.Size([2, 2, 2])
  >>> y = torch.squeeze(x, 0)
  >>> y.size()
  torch.Size([2, 1, 2, 1, 2])
  >>> y = torch.squeeze(x, 1) # specify dimension
  >>> y.size()
  torch.Size([2, 2, 1, 2])
  ```

### `unsqueeze`

```python
torch.unsqueeze(input, dim) → Tensor
```

- Documentation: [`torch.unsqueeze`](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)

- Returns a new tensor with a dimension of size one inserted at the specified position.

- Example

  ```python
  x = torch.tensor([1, 2, 3, 4])
  x.shape
  ```

  ```txt
  torch.Size([4])
  ```

  ```python
  torch.unsqueeze(x, 0), torch.unsqueeze(x, 0).shape
  ```

  ```txt
  (tensor([[1, 2, 3, 4]]), torch.Size([1, 4]))
  ```

  ```python
  torch.unsqueeze(x, 1), torch.unsqueeze(x, 1).shape
  ```

  ```txt
  (tensor([[1],
           [2],
           [3],
           [4]]), torch.Size([4, 1]))
  ```

We can also achieve the same effect with `view`:

```python
y = x.view(-1, 4) # same as torch.unsqueeze(x, 0)
y, y.shape
```

```txt
(tensor([[1, 2, 3, 4]]), torch.Size([1, 4]))
```

```python
y = x.view(4, -1) # same as torch.unsqueeze(x, 1)
y, y.shape
```

```txt
(tensor([[1],
         [2],
         [3],
         [4]]), torch.Size([4, 1]))
```

`unsqueeze()` is particularly useful when we feed a **single** sample to our neural network. PyTorch requires mini-batch input, let's say dimension is `[batch_size, channels, w, h]`. However, a single sample has the dimension `[channels, w, h]`, which could lead to dimension error when we feed it to the network. We can use `unsqueeze()` to change its dimension to `[1, channels, w, h]`. This mocks up a mini-batch with only one sample and won't cause any error.



## Casting Tensor's Type

Casting in PyTorch is as simple as typing the name of the type you wish to cast to, and treating it as a method.

Example

```python
x = torch.ones([2, 3], dtype=torch.int32)
x
```

```txt
tensor([[1, 1, 1],
        [1, 1, 1]], dtype=torch.int32)
```

```python
y = x.float() # cast to float
y, y.dtype
```

```txt
(tensor([[1., 1., 1.],
         [1., 1., 1.]]), torch.float32)
```



## Require Gradient

To track the gradient of a tensor, its `requires_grad` attribute should be `True`. There're two ways to achieve this:

1. Create a tensor, then call `requires_grad_()`

   Example:

   ```python
   import torch
   
   x = torch.ones(2, 3)
   x
   ```

   ```txt
   tensor([[1., 1., 1.],
           [1., 1., 1.]])
   ```

   ```python
   x.requires_grad_()
   x
   ```

   ```txt
   tensor([[1., 1., 1.],
           [1., 1., 1.]], requires_grad=True)
   ```

2. Specify `requires_grad=True` during creating tensor

   ```python
   y = torch.ones((2, 3), requires_grad=True)
   y
   ```

   ```txt
   tensor([[1., 1., 1.],
           [1., 1., 1.]], requires_grad=True)
   ```

   

## Not Taking Gradient

During training, we will update the weights and biases based on the gradient and learning rate. 

When we do so, we have to tell PyTorch not to take the gradient of this step too—otherwise things will get very confusing when we try to compute the derivative at the next batch! 

### Assign to tensor's `data` attribute

If we assign to the `data` attribute of a tensor then PyTorch will not take the gradient of that step.

E.g.

```python
for p in model.parameters():
  p.data -= p.grad * lr
```

### Wrap in `torch.no_grad()`

E.g.

```python
with torch.no_grad():
	for p in models.parameters():
    p -= p.grad * lr
```

## Enable/Disabel gradient tracking dynamically

Use `torch.set_grad_enabled (mode: bool)` ([Documentation](https://pytorch.org/docs/stable/generated/torch.set_grad_enabled.html))

```python
with torch.set_grad_enabled(flag):
    // do something
    pass
```



## Zero the Gradient

In PyToch, after updating the parameters using their gradients in one iteration, the gradients need to be zeroed. Otherwise they'll be accumulated in the next iteration!

### `zero_()`

E.g.

```python
for p in model.parameters():
  p.grad.zero_()
```

### Set `grad` as None

E.g.

```python
for p in model.parameters():
  p.grad = None
```

### Use `optimizer.zero_grad()`



## Running loss

 `loss.item()` returns the average loss for each sample within the batch, i.e. loss of entire mini-batch but divided by the batch size. Therefore, to get the running loss of the mini-batch, we need to do:

```python
running_loss += loss.item() * batch_size
```

Reference

- [What is running loss in PyTorch and how is it calculated](https://stackoverflow.com/questions/61092523/what-is-running-loss-in-pytorch-and-how-is-it-calculated)

- [How to calculate running loss/training loss while training a CNN model](https://discuss.pytorch.org/t/how-to-calculate-running-loss-training-loss-while-training-a-cnn-model/49301)



## Number of model's parameters

### Use `model.parameters()`

Sum the number of elements for every parameter group:

```python
def get_num_params(model):
    return sum(p.numel() for p in model.parameters())
```

Calculate only the *trainable* parameters:

```python
def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Reference: [How do I check the number of parameters of a model?](https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325)

### Use [`model.named_paramters()`](https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.named_parameters) 

To get the parameter count of each layer, PyTorch has [model.named_paramters()](https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.named_parameters) that returns an iterator of both the parameter name and the parameter itself.

Example

```python
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
            
    print(table)
    print(f"Total Trainable Params: {total_params}")
    
    return total_params
    
count_parameters(net)
```

Reference: [Check the total number of parameters in a PyTorch model](https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model)

## Split tensors

- [`torch.split()`](https://pytorch.org/docs/stable/generated/torch.split.html): Splits the tensor into chunks of specified sizes
- [`torch.chunk()`](https://pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk): Attempts to split a tensor into the specified number of chunks.

### `torch.split()`

Split tensor evenly:

```python
import torch

tensor = torch.rand((4, 6, 2, 2))
tensor.shape
```

```txt
torch.Size([4, 6, 2, 2])
```

```python
tensor_split = tensor.split(1, dim=1) # split tensor into chunks of size 1 along dimension 1
print(f"#chunks: {len(tensor_split)}; chunk size: {tensor_split[0].shape}")
```

```txt
#chunks: 6; chunk size: torch.Size([4, 1, 2, 2])
```

You can also split the tensor into chunks of different specific sizes (i.e. not evenly):

```python
chunk_sizes = [1, 2, 3]
tensor_split = tensor.split(chunk_sizes, dim=1) # split a into chunks of size 1 along dimension 1
print(f"#chunks: {len(tensor_split)}")
for idx, chunk in enumerate(tensor_split):
    print(f"Shape of chunk {idx}: {chunk.shape}")
```

```txt
#chunks: 3
Shape of chunk 0: torch.Size([4, 1, 2, 2])
Shape of chunk 1: torch.Size([4, 2, 2, 2])
Shape of chunk 2: torch.Size([4, 3, 2, 2])
```

### `torch.chunk()`

Split a tensor into the specified number of chunks

```python
tensor_chunks = tensor.chunk(3, dim=1) # split tensor into 3 chunks along dimension 1
print(f"#chunks: {len(tensor_chunks)}")
for idx, chunk in enumerate(tensor_chunks):
    print(f"Shape of chunk {idx}: {chunk.shape}")
```

```txt
#chunks: 3
Shape of chunk 0: torch.Size([4, 2, 2, 2])
Shape of chunk 1: torch.Size([4, 2, 2, 2])
Shape of chunk 2: torch.Size([4, 2, 2, 2])
```

