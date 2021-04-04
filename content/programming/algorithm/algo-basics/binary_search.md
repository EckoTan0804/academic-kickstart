---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 102

# Basic metadata
title: "Binary Search"
date: 2021-04-04
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Algo", "Basics", "Binary Search"]
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
        weight: 2
---

Binary search only works when your list is in **sorted order**.

## Code

```python
def binary_search(list, item):
    # keep track of low and high
    low = 0
    high = len(list) - 1

    while low <= high: # when we haven't narrowed down to one element
        mid = (low + high) // 2 # index of the middle element
        guess = list[mid] # value of the middle element

        if guess == item: 
            return mid
        elif guess > item: 
            # Guess is too high,
            # we narrow our search on the "left" part
            high = mid - 1
        else:
            # Guess is too low,
            # we narrow our search on the "right" part
            low = mid + 1

    # If we have anrrowed down to one element but still can not find the item
    # => Item is not in the list 
    return -1
```

## Complexity

- Time complexity: $O(\log n)$

## Binary Vs Linear Search

### Average Case

![Binary vs Linear Search Linear Search](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/binary-and-linear-search-animations.gif)

### Binary Search Worst Case

![linear-vs-binary-search-worst-case](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/linear-vs-binary-search-worst-case.gif)

### Binary Search Best Case

![linear-vs-binary-search-best-case](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/linear-vs-binary-search-best-case.gif)