---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 1101

# Basic metadata
title: "Model Registration"
date: 2022-03-09
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "PyTorch Issues"]
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
        parent: pytorch-issues
        weight: 1
---

Before training the model, modules that need to be trained must be correctly registered. Otherwise, the unregistered modules would NOT be trained without errors or exceptions being thrown. Moreover, when we call `model.cuda()`, the unregistered modules will stay on CPU and will not be moved to GPU. In other words, this gotcha is usually hard to notice.

## When does this gotcha usually occurs?

1. Use python's `list` or  `dict`but forget to wrap it with [`nn.ModuleList`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) or [`nn.ModuleDict`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html).
   - In this case, PyTorch can not correctly recognize its elements as trainable modules. Therefore, they can NOT be correctly registered and trained.
2. An attribute of the model is python's `list` or  `dict`, but forget to wrap it with [`nn.ModuleList`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) or [`nn.ModuleDict`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html).

### Example

```python
import torch
import torch.nn as nn

class DummyModule(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print("dummy")
        return x
```

```python
class Net(nn.Module):

    def __init__(self, num_dummy_modules=4):
        super().__init__()
        # Here self.dummy_modul_list is just a python list, 
        # as we do not wrap it with nn.ModuleList
        self.dummy_module_list = [DummyModule().cuda() for _ in range(num_dummy_modules)]
        print(f"#dummy modules: {len(self.dummy_module_list)}")

    def forward(self, x):
        for dummy_module in self.dummy_module_list:
            x = dummy_module(x)
        return x
```

Now we initialize the model and move it to GPU:

```python
model = Net().to(device)
print(model)
```

```txt
#dummy modules: 4
Net()
```

We can see that `Net` contains nothing. The 4 `DummyModule` are not registered.

Now we use `nn.ModuleList` to wrap `self.dummy_modul_list` and covert its element to registered trainable modules.

```python
class Net(nn.Module):

    def __init__(self, num_dummy_modules=4):
        super().__init__()
        self.dummy_module_list = [DummyModule().cuda() for _ in range(num_dummy_modules)]
        # Register elements in self.dummy_module_list as trainable modules
        self.dummy_module_list = nn.ModuleList(self.dummy_module_list)
        print(f"#dummy modules: {len(self.dummy_module_list)}")

    def forward(self, x):
        for dummy_module in self.dummy_module_list:
            x = dummy_module(x)
        return x
```

```python
model = Net().to(device)
print(model)
```

```txt
#dummy modules: 4
Net(
  (dummy_module_list): ModuleList(
    (0): DummyModule()
    (1): DummyModule()
    (2): DummyModule()
    (3): DummyModule()
  )
)
```



## References

- [网络模型构建](https://hellojialee.github.io/2020/05/28/Pytorch%E5%AE%9E%E7%94%A8%E6%8C%87%E5%8D%97/)

