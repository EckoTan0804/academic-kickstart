---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 903

# Basic metadata
title: "nn ModuleList vs. Sequential"
date: 2020-11-09
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
        weight: 3
---

```python
import torch 
import torch.nn as nn
import torch.nn.functional as F
```

## `nn.Module`

- Defines the base class for all neural network
- We MUST *subclass* it

### Example

```python
class Net(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 28 * 28) # flat
        
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        
        return x
```

```python
model = Net(1, 10)
model
```

```
Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=25088, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=10, bias=True)
)
```

## `nn.Sequential`

[Sequential](https://pytorch.org/docs/stable/nn.html?highlight=sequential#torch.nn.Sequential) is a container of Modules that can be stacked together and run at the same time.

- The `nn.Module`'s stored in `nn.Sequential` are connected in a cascaded way
- `nn.Sequential` has a `forward()` method
  - Have to make sure that the output size of a block matches the input size of the following block.
- Basically, it behaves just like a `nn.Module`

### Example

```python
class NetSequential(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(-1, 32 * 28 * 28)
        x = self.decode(x)
        return x
```

```python
model = NetSequential(1, 10)
model
```

```
NetSequential(
  (conv_block1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (conv_block2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (decoder): Sequential(
    (0): Linear(in_features=25088, out_features=1024, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=1024, out_features=10, bias=True)
  )
)
```

## `nn.ModuleList`

> [Documentation](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html):
>
> Holds submodules in a list.
>
> [`ModuleList`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList) can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all[`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) methods.

- Does NOT have a `forward()` method, because it does not define any neural network, that is, there is no connection between each of the `nn.Module`'s that it stores.
- We may use it to store `nn.Module`'s, just like you use Python lists to store other types of objects (integers, strings, etc). And Pytorch is “aware” of the existence of the `nn.Module`'s inside an `nn.ModuleList`
- Execution order of `nn.Modules` stored in `nn.ModuleList` is defined in `forward()`, which we have to implement explicitly by ourselves.

### Example

```python
class NetModuleList(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        ])

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x
```

```python
model = NetModuleList(1, 10)
model
```

```
NetModuleList(
  (module_list): ModuleList(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=25088, out_features=1024, bias=True)
    (8): Sigmoid()
    (9): Linear(in_features=1024, out_features=10, bias=True)
  )
)
```

## `nn.Sequential` vs. `nn.ModuleList`

|                                                | `nn.Sequential` | `nn.ModuleList` |
| ---------------------------------------------- | --------------- | --------------- |
| Has `forward()` ?                              | ✅               | ❌               |
| Connection between `nn.Modules` stored inside? | ✅               | ❌               |
| Execution order = stored order?                | ✅               | ❌               |
| Advantages                                     | succinct        | flexible        |

### When to use which?

- Use `Module` when we have a big block compose of multiple smaller blocks
- Use `Sequential` when we want to create a small block from layers
- Use `ModuleList` when we need to iterate through some layers or building blocks and do something

## Reference

- [When should I use nn.ModuleList and when should I use nn.Sequential?](https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463)

- [Pytorch: how and when to use Module, Sequential, ModuleList and ModuleDict](https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17)
- [PyTorch 中的 ModuleList 和 Sequential: 区别和使用场景](https://zhuanlan.zhihu.com/p/64990232)