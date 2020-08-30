---
# Basic info
title: "Get Dictionary Items with Specified Initialization"
linktitle: ""
date: 2020-07-08
draft: false
# type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics"]
categories: ["Coding"]
toc: true # Show table of contents?

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: "Examples for conditionally changing values in Python list"
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
---

## **`dict.get(key, default = None))`**

returns the value of the item with the specified key.

Parameters

- `key` − This is the Key to be searched in the dictionary.
- `default` − This is the Value to be returned in case key does not exist.

See: [Python dictionary `get()` Method](https://www.tutorialspoint.com/python/dictionary_get.htm)

We can use `dict.get(key, init_value)` for initializing key-value pair in dictionary

E.g.: Counting how many times each element occurs in a list

```python
 a = ["a", "b", "a", "a", "c", "c","a"]
 d = {}
 for el in a:
     d[el] = d.get(el, 0) + 1
```

```python
d
```

```
 {'a': 4, 'b': 1, 'c': 2}
```