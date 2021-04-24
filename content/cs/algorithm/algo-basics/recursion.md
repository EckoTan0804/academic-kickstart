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

## Divide and Conquer 

**Divide and conquer (D&C)** is a well-known recursive technique for solving problems. To solve a problem using D&C, there're 2 steps:

1. **Figure out the base case.** This should be the simplest possible case.
2. **Divide or decrease the problem until it becomes the base case.**

### Sum of Array

Let's calculate the sum of an array using D&C. 

1. Figure out the  base case.

   Base case is that the array is empty. In this situation, sum of this array is 0.

2. Decrease the problem until it becomes the base case.

   ```
   sum = a[0] + a[1] + a[2]
   	= a[0] + (a[1] + a[2])
   	= a[0] + (a[1] + (a[2] + 0))
   ```

Example:

![截屏2021-04-06 23.36.21](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-04-06%2023.36.21.png)

As we can see, in the 2nd and 3rd step, we're passing a smaller (sub)array into the `sum` function. That is, we're decreasing the size of our problem!

When we have figured out the two steps of D&C, the implementation is pretty simple:

```python
def sum(arr):
    if not arr: # arr is empty
        return 0
    else:
        return arr[0] + sum(arr[1:])
```

{{% alert note %}} 

When you’re writing a recursive function involving an array, the base case is often an **empty array** or an **array with one element**. If you’re stuck, try that first.

{{% /alert %}}

### Count Number of Elements

```python
def length(arr):
    if not arr:
        # base case: length of empty array is 0
        return 0
    
    # recursive case: 1 + length of the remaining subarray
    return 1 + length(arr[1:])
```

### Find Maximum 

```python
def max(arr):
    if len(arr) == 2:
        return arr[0] if arr[0] > arr[1] else arr[1]
    return arr[0] if arr[0] > max(arr[1:]) else max(arr[1:])
```

### Recursive Binary Search

```python
def binary_search(arr, low, high, target):
    if low <= high:
        mid = (low + high) // 2
        if arr[mid] > target:
            # If element is smaller than mid, 
            # then it can only be present in left subarray
            return binary_search(arr, low, mid-1, target)
        elif arr[mid] == target:
            return mid
        else:
            # If element is greater than mid, 
            # then it can only be present in right subarray
            return binary_search(arr, mid+1, high, target)

    else: # Element is not present in the array
        return -1
```

