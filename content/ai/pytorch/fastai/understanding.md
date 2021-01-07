---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 110

# Basic metadata
title: "fastai Understanding"
date: 2020-09-25
draft: true
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "fastai"]
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
        parent: fastai
        weight: 1
---

## `ls`

We can see what's in this directory by using `ls`, a method added by fastai. 

- This method returns an object of a special fastai class called `L`, which has all the same functionality of Python's built-in `list`, plus a lot more. 
- One of its handy features is that, when printed, it displays the count of items, before listing the items themselves (if there are more than 10 items, it just shows the first few)

Example:

```python
from fastai.vision.all import *

path = untar_data(URLs.MNIST_SAMPLE)
path.ls()
```

```txt
(#9) [Path('cleaned.csv'),Path('item_list.txt'),Path('trained_model.pkl'),Path('models'),Path('valid'),Path('labels.csv'),Path('export.pkl'),Path('history.csv'),Path('train')]
```

