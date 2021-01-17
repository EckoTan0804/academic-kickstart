---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 906

# Basic metadata
title: "Saving and Loading Models"
date: 2021-01-17
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "PyTorch Recipe"]
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
        parent: pytorch-recipes
        weight: 6
---

Three core functions for saving and loading models:

1. [torch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save)

   Saves a serialized object to disk. This function uses Python’s [pickle](https://docs.python.org/3/library/pickle.html) utility for serialization. Models, tensors, and dictionaries of all kinds of objects can be saved using this function.

2. [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load)

   Uses [pickle](https://docs.python.org/3/library/pickle.html)’s unpickling facilities to deserialize pickled object files to memory. 

3. [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/nn.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)

   Loads a model’s parameter dictionary using a deserialized *state_dict*. 

## `state_dict`

In PyTorch, 

- the learnable parameters (i.e. weights and biases) of an `torch.nn.Module` model are contained in the model’s **parameters** (accessed with `model.parameters()`). A **state_dict** is simply a Python **dictionary object** that maps each layer to its parameter tensor. 
  - Note that only layers with **learnable** parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s running_mean) have entries in the model’s **state_dict**. 

- Optimizer objects (`torch.optim`) also have a **state_dict**, which contains information about the optimizer’s state, as well as the hyperparameters used.

Because **state_dict** objects are Python dictionaries, they can be easily saved, updated, altered, and restored, adding a great deal of modularity to PyTorch models and optimizers.

### Example

```python
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

```python
# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

```python
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
```

```
Model's state_dict:
conv1.weight 	 torch.Size([6, 3, 5, 5])
conv1.bias 	 torch.Size([6])
conv2.weight 	 torch.Size([16, 6, 5, 5])
conv2.bias 	 torch.Size([16])
fc1.weight 	 torch.Size([120, 400])
fc1.bias 	 torch.Size([120])
fc2.weight 	 torch.Size([84, 120])
fc2.bias 	 torch.Size([84])
fc3.weight 	 torch.Size([10, 84])
fc3.bias 	 torch.Size([10])
```

```python
# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

```
Optimizer's state_dict:
state 	 {}
param_groups 	 [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]
```

## Saving & Loading Model for Inference

### Save/Load `state_dict` (Recommended)

**Save**:

```python
torch.save(model.state_dict(), PATH)
```

**Load**:

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

When saving a model for inference, it is only necessary to save the trained model’s learned parameters. Saving the model’s *state_dict* with the `torch.save()` function will give you the most flexibility for restoring the model later.

A common PyTorch convention is to save models using either a `.pt` or `.pth` file extension.

### Save/Load entire model

**Save:**

```python
torch.save(model, PATH)
```

**Load:**

```python
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()
```

This save/load process uses the most intuitive syntax and involves the least amount of code. Saving a model in this way will save the **entire** module using Python’s [pickle](https://docs.python.org/3/library/pickle.html) module.

🔴 The disadvantage of this approach is that the serialized data is bound to the specific classes and the exact directory structure used when the model is saved. The reason for this is because pickle does not save the model class itself. Rather, it saves a path to the file containing the class, which is used during load time. Because of this, your code can break in various ways when used in other projects or after refactors.

A common PyTorch convention is to save models using either a `.pt` or `.pth` file extension.

## Saving & Loading a General Checkpoint for Inference and/or Resuming Training

See: [Saving and Loading Checkpoints]({{< relref "saving-and-loading-checkpoints.md" >}})

## Reference

- [SAVING AND LOADING MODELS](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-and-loading-models)

- [What is the difference between .pt, .pth and .pwf extentions in PyTorch?](https://stackoverflow.com/questions/59095824/what-is-the-difference-between-pt-pth-and-pwf-extentions-in-pytorch)

