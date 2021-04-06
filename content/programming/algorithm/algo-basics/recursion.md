---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 104

# Basic metadata
title: "Recursion"
date: 2021-04-06
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Algo", "Basics", "Recursion"]
categories: ["Coding"]
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
    algo:
        parent: algo-basics
        weight: 4
---

Every recursive function has two parts

- **base case**: the function does not call itself again, so it will not go into an infinite loop
- **recursive case**: the function calls itself

Sometimes recursion could be also re-written using loops. In fact, loops are sometimes   better for performance. But recursion is used when it makes the solution clearer.

## Examples

### Count down

```python
def countdown(i):
    print(i) 
    if i <= 0: # base case
        return
    else: # recursive case
        countdown(i-1)
```

```python
countdown(5)
```

```txt
5
4
3
2
1
0
```

![截屏2021-04-06 18.41.21](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-04-06%2018.41.21.png)

### Factorial

`factorial(3)` is defined as $3! = 3 \cdot 2 \cdot 1 = 3$

```python
def factorial(x):
    # base case
    if x == 1:
        return 1 # 1! = 1
    else:
        return x * factorial(x-1)
```

```python
factorial(3)
```

```txt
6
```

```python
factorial(5)
```

```txt
120
```

