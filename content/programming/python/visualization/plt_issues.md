---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 503

# Basic metadata
title: "Matplotlib Issues"
date: 2022-01-23
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Visualization", "Matploblib"]
categories: ["coding"]
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
    python:
        parent: visualization
        weight: 3

---

Just to mark down some issues I have met when using `matplotlib.pyplot` for plotting. 

## Change global font to Times New Roman

```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
```

Source: [Matplotlib cannot find basic fonts](https://stackoverflow.com/a/66462451/4891826)

