---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 1005

# Basic metadata
title: "YACS for Configuration Management"
date: 2021-05-12
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "NN Training"]
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
        parent: training
        weight: 5
---

**YACS** stands for Yet Another Configuration System. It helps define and manage system configurations such as hyperparameters and architecture/module choices for training a model. A tool like this one is essential to reproducibility and is a fundamental component of the system.

## Usage

YACS can be used in a variety of flexible ways. There are two main paradigms:

- Configuration as *local variable* (recommend)
- Configuration as a *global singleton*

### Step 1: Create Project Config File

- Create a folder called `config`

- In `config` folder, create config file, typically called `config.py` or `default.py`

  ```
  my_project
  |- config
  	|- __init__.py
  	|- default.py
  |- experiment.yaml
  |- main.py
  ```

  This file is the one-stop reference point for all configurable options. It should be very well documented and provide sensible defaults for all options. 

  E.g.

  ```python
  # my_project/config/default.py
  
  from yacs.config import CfgNode as CN
  
  
  _C = CN()
  
  _C.SYSTEM = CN()
  # Number of GPUS to use in the experiment
  _C.SYSTEM.NUM_GPUS = 8
  # Number of workers for doing things
  _C.SYSTEM.NUM_WORKERS = 4
  
  _C.TRAIN = CN()
  # A very important hyperparameter
  _C.TRAIN.HYPERPARAMETER_1 = 0.1
  # The all important scales for the stuff
  _C.TRAIN.SCALES = (2, 4, 8, 16)
  
  
  def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
  
  # Alternatively, provide a way to import the defaults as a global singleton
  # cfg = _C  # users can `from config import cfg`
  ```

> For global singleton, another way is to declare `cfg` in `__init__.py`:
>
> ```python
> # my_project/config/__init__.py
> 
> from .default import _C as cfg
> ```
>
> Then, in other script, we can import it like this:
>
> ```python
> from config import cfg
> ```

### Step 2: Create YAML Configuration Files

Typically you'll make one for each experiment. Each configuration file only overrides the options that are changing in that experiment.

```yaml
# my_project/experiment.yaml

SYSTEM:
  NUM_GPUS: 2
TRAIN:
  SCALES: (1, 2)
```

### Step 3: Use Config in Actual Project Code

Local variable usage pattern:

```python
# my_project/main.py

from config import get_cfg_defaults 

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("experiment.yaml")

    # Override from a list
    opts = ["SYSTEM.NUM_WORKERS", 8]
    cfg.merge_from_list(opts)

    # Freeze the config to prevent further modification
    cfg.freeze()
    print(cfg)
    
    # Further code using config settings
    # ...
```

```txt
SYSTEM:
  NUM_GPUS: 2
  NUM_WORKERS: 8
TRAIN:
  HYPERPARAMETER_1: 0.1
  SCALES: (1, 2)
```

We can also use global singleton usage pattern:

```python
from config import cfg

if __name__ == "__main__":
    cfg.merge_from_file("experiment.yaml")

    # Override from a list
    opts = ["SYSTEM.NUM_WORKERS", 8]
    cfg.merge_from_list(opts)

    # Freeze the config to prevent further modification
    cfg.freeze()
    print(cfg)
    
    # Further code using config settings
    # ...
```

## Reference

- [rbgirshick](https://github.com/rbgirshick)/**[yacs](https://github.com/rbgirshick/yacs)**