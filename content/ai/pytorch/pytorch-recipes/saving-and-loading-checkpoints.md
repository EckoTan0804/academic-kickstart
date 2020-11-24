---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 902

# Basic metadata
title: "Saving and Loading Checkpoints"
date: 2020-11-06
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
        weight: 2
---

## Motivation

Saving and loading a general checkpoint model for inference or resuming training can be helpful for picking up where we last left off. 

When saving a general checkpoint, you must save more than just the model’s `state_dict.` It is important to also save the optimizer’s `state_dict`, as this contains buffers and parameters that are updated as the model trains. Other items that you may want to save are the 

- epoch you left off on, 
- the latest recorded training loss, 
- external `torch.nn.Embedding` layers, 
- and more, based on your own algorithm.

## How to save and load checkpoints?

To **save** multiple checkpoints, we must organize them in a dictionary and use `torch.save()` to serialize the dictionary. A common PyTorch convention is to save these checkpoints using the `.tar` file extension. 

To **load** the items, 

1. first initialize the model and optimizer, 
2. then load the dictionary locally using `torch.load()`. From here, we can easily access the saved items by simply querying the dictionary as you would expect.

## Example

### 1. Import necessary libraries for loading our data

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

### 2. Define and intialize the neural network

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

net = Net()
```

### 3. Initialize the optimizer

```python
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

### 4. Saving the general checkpoint

1. Collect all relevant information, 
2. Build our checkpoint `dictionary`.
3. Save checkpoint using `torch.save()`

```python
# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4 # just dummy number

torch.save({'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS
            }, PATH)
```

## 5. Load the general checkpoint

1. First initialize the model and optimizer
2. Then load the checkpoint `dictionary` locally

```python
# initialize the model and optimizer
model = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# load checkpoint
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

3. Call `eval()` for inference or `train()` for training



## Google Colab Notebook

[Colab Notebook](https://colab.research.google.com/drive/1PlsftZnPEvyWkJUXIoM5M3a-UA1RTXhl?authuser=1) 







## Reference

- [SAVING AND LOADING A GENERAL CHECKPOINT IN PYTORCH](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)


