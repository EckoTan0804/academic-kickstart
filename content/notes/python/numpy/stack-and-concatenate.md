---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 301

# Basic metadata
title: "Stack and Concatenate"
date: 2020-08-16
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
        weight: 1

---

## TL;DR

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/np_array-stack.png" title="Stack of Numpy array" numbered="true" >}}

## Prepare Data

```python
import numpy as np
```


```python
x1 = np.array([[[9, 3, 7, 3],
        [2, 1, 1, 2],
        [1, 4, 2, 5]],

       [[5, 5, 2, 5],
        [7, 7, 6, 1],
        [6, 7, 2, 3]]])
```


```python
x1
```


    array([[[9, 3, 7, 3],
            [2, 1, 1, 2],
            [1, 4, 2, 5]],
    
           [[5, 5, 2, 5],
            [7, 7, 6, 1],
            [6, 7, 2, 3]]])


```python
x2 = np.random.randint(20, size=(2, 3, 4))
x2
```


    array([[[16, 13, 19, 16],
            [15, 19,  1,  2],
            [ 4, 15,  5,  7]],
    
           [[18, 18,  1,  9],
            [ 5, 19,  7, 11],
            [ 8, 10,  2, 18]]])

## `np.hstack()`


```python
np.hstack((x1, x2)) # horizontally stack, equivalent to concatenate along column
```


    array([[[ 9,  3,  7,  3],
            [ 2,  1,  1,  2],
            [ 1,  4,  2,  5],
            [16, 13, 19, 16],
            [15, 19,  1,  2],
            [ 4, 15,  5,  7]],
    
           [[ 5,  5,  2,  5],
            [ 7,  7,  6,  1],
            [ 6,  7,  2,  3],
            [18, 18,  1,  9],
            [ 5, 19,  7, 11],
            [ 8, 10,  2, 18]]])

**Equivalent to `np.concatenate` with parameter `axis=1` (along column)**


```python
np.concatenate([x1, x2], axis=1) # concatenate along column
```


    array([[[ 9,  3,  7,  3],
            [ 2,  1,  1,  2],
            [ 1,  4,  2,  5],
            [16, 13, 19, 16],
            [15, 19,  1,  2],
            [ 4, 15,  5,  7]],
    
           [[ 5,  5,  2,  5],
            [ 7,  7,  6,  1],
            [ 6,  7,  2,  3],
            [18, 18,  1,  9],
            [ 5, 19,  7, 11],
            [ 8, 10,  2, 18]]])

## `np.vstack()`


```python
np.vstack([x1, x2]) # vertically stack, equivalent to concatenate along row
```


    array([[[ 9,  3,  7,  3],
            [ 2,  1,  1,  2],
            [ 1,  4,  2,  5]],
    
           [[ 5,  5,  2,  5],
            [ 7,  7,  6,  1],
            [ 6,  7,  2,  3]],
    
           [[16, 13, 19, 16],
            [15, 19,  1,  2],
            [ 4, 15,  5,  7]],
    
           [[18, 18,  1,  9],
            [ 5, 19,  7, 11],
            [ 8, 10,  2, 18]]])

**Equivalent to `np.concatenate` with parameter `axis=0` (along row)** 


```python
np.concatenate([x1, x2], axis=0) # concatenate along row
```


    array([[[ 9,  3,  7,  3],
            [ 2,  1,  1,  2],
            [ 1,  4,  2,  5]],
    
           [[ 5,  5,  2,  5],
            [ 7,  7,  6,  1],
            [ 6,  7,  2,  3]],
    
           [[16, 13, 19, 16],
            [15, 19,  1,  2],
            [ 4, 15,  5,  7]],
    
           [[18, 18,  1,  9],
            [ 5, 19,  7, 11],
            [ 8, 10,  2, 18]]])




```python
np.vstack([x1, x2]).shape 
```


    (4, 3, 4)

Note: for 1-D array of shape `(N,)`, the array will be firstly reshape to `(1, N)`

```python
a = np.arange(3) # [0, 1, 2], shape: (3, )
b = np.arange(4, 7) # [4, 5, 6], shape: (3, )

print(np.vstack([a, b]))
print('shape:', np.vstack([a, b]).shape)
```

```
[[0 1 2]
 [4 5 6]]
shape: (2, 3)
```



## `np.dstack()`


```python
np.dstack([x1, x2]) # depth-wise stack, equivalent to concatenate along the third axis (depth)
```


    array([[[ 9,  3,  7,  3, 16, 13, 19, 16],
            [ 2,  1,  1,  2, 15, 19,  1,  2],
            [ 1,  4,  2,  5,  4, 15,  5,  7]],
    
           [[ 5,  5,  2,  5, 18, 18,  1,  9],
            [ 7,  7,  6,  1,  5, 19,  7, 11],
            [ 6,  7,  2,  3,  8, 10,  2, 18]]])

**Equivalent to `np.concatenate` with parameter `axis=2` (along depth)** 


```python
np.concatenate([x1, x2], axis=2) # concatenate along the third axis
```


    array([[[ 9,  3,  7,  3, 16, 13, 19, 16],
            [ 2,  1,  1,  2, 15, 19,  1,  2],
            [ 1,  4,  2,  5,  4, 15,  5,  7]],
    
           [[ 5,  5,  2,  5, 18, 18,  1,  9],
            [ 7,  7,  6,  1,  5, 19,  7, 11],
            [ 6,  7,  2,  3,  8, 10,  2, 18]]])