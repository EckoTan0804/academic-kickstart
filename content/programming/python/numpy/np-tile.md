---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 304

# Basic metadata
title: "Numpy Tile"
date: 2020-11-22
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Numpy"]
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
        parent: numpy
        weight: 4
---

```python
x = np.array([[1, 2],
              [3, 4]])
print(x)
print(x.shape)
```

```
[[1 2]
 [3 4]]
(2, 2)
```

```python
x1 = np.tile(x, (1, 2))
print(x1)
print(x1.shape)
```

```
[[1 2 1 2]
 [3 4 3 4]]
(2, 4)
```

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/np-tile.png" title="Numpy tile" numbered="true" >}}

## Reference

- [numpy tile documentation](https://numpy.org/doc/stable/reference/generated/numpy.tile.html)

- {{< youtube 9BjmmK61pjI >}}