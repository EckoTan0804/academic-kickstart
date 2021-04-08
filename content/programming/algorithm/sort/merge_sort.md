---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 303

# Basic metadata
title: "Merge Sort"
date: 2021-04-06
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Algo", "Sort"]
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
        parent: sort
        weight: 3
---

## TL;DR

- Merge sort use Divide and Conquer strategy
  - Repeatedly divides the array into two subarrays until subarrays of size 1
  - Merge two sorted array to achieve a bigger sorted array
- Time complexity is $O(n \log n)$
- Space complexity is $O(n)$ (non-inplace)

## How Merge Sort Works?

### Divide and Conquer Strategy

Merge Sort is one of the most popular sorting algorithms that is based on the principle of [Divide and Conquer](https://www.programiz.com/dsa/divide-and-conquer).

Using the **Divide and Conquer** technique, we divide a problem into subproblems. When the solution to each subproblem is ready, we "combine" the results from the subproblems to solve the main problem.

Suppose we had to sort an array `arr`. A subproblem would be to sort a sub-section of this array starting at index `p` and ending at index `r`, denoted as `arr[p..r]`.

- **Divide**

  If `q` is the half-way point between `p` and `r`, then we can split the subarray `arr[p..r]` into two arrays `arr[p..q]` and `arr[q+1, r]`.

- **Conquer**

  In the conquer step, we try to sort both the subarrays `arr[p..q]` and `arr[q+1, r]`. If we haven't yet reached the base case, we again divide both these subarrays and try to sort them.

- **Combine**

  When the When the conquer step reaches the base step and we get two sorted subarrays `arr[p..q]` and `arr[q+1, r]` for array `arr[p..r]`, we combine the results by creating a sorted array `arr[p..r]` from two sorted subarrays `arr[p..q]` and `arr[q+1, r]`

### The MergeSort Algorithm

- Repeatedly divides the array into two halves until we reach a stage where we try to perform MergeSort on a subarray of size 1 (the base case)
- Combine the sorted arrays into larger sorted arrays until the whole array is merged.

## Implementation

**Merge**: merge two sorted arrays to a larger sorted array

```python
def merge(left_arr, right_arr, arr):
    """
    Merge left_arr and right_arr to arr.
    left_arr and right_arr are already sorted.
    After merging, arr should be sorted.
    """
    left_len = len(left_arr)
    right_len = len(right_arr)

    i = 0 # Mark the position of current smallest element in left_arr
    j = 0 # Mark the position of current smallest element in right_arr
    k = 0 # Mark the position to be filled in arr

    while (i < left_len and j < right_len): # left_arr and right_arr are not exhausted
        # We compare the current smallest elements of left_arr and right_arr,
        # fill arr with the smaller one
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1

        k += 1
    
    if i > left_len - 1: 
        # left_arr is exhausted.
        # Thus, we fill up the rest of arr with right_arr
        arr[k:] = right_arr[j:]
    if j > right_len - 1: \
        # right_arr is exhausted
        # Thus, we fill up the rest of arr with left_arr
        arr[k:] = left_arr[i:]
    
    return arr
```

```python
arr = [2, 4, 1, 6, 8, 5, 3, 7]
left_arr = [1, 2, 4, 6]
right_arr = [3, 5, 7, 8]
merge(left_arr, right_arr, arr)
```

```
[1, 2, 3, 4, 5, 6, 7, 8]
```

![merge_sort-merge](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/merge_sort-merge.png)

**Merge sort**

```python
def merge_sort(arr):
    print(arr)
    
    # Base case
    if len(arr) < 2:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left_arr = arr[:mid]
    right_arr = arr[mid:]

    # Conquer
    merge_sort(left_arr)
    merge_sort(right_arr)
    
    # Combine
    merge(left_arr, right_arr, arr)
    print(f"Merge {left_arr} and {right_arr} --> {arr}\n")

    return arr

```

```python
arr = [2, 4, 1, 6, 8, 5, 3, 7]
print(merge_sort(arr))
```

```txt
[2, 4, 1, 6, 8, 5, 3, 7]
[2, 4, 1, 6]
[2, 4]
[2]
[4]
Merge [2] and [4] --> [2, 4]

[1, 6]
[1]
[6]
Merge [1] and [6] --> [1, 6]

Merge [2, 4] and [1, 6] --> [1, 2, 4, 6]

[8, 5, 3, 7]
[8, 5]
[8]
[5]
Merge [8] and [5] --> [5, 8]

[3, 7]
[3]
[7]
Merge [3] and [7] --> [3, 7]

Merge [5, 8] and [3, 7] --> [3, 5, 7, 8]

Merge [1, 2, 4, 6] and [3, 5, 7, 8] --> [1, 2, 3, 4, 5, 6, 7, 8]

[1, 2, 3, 4, 5, 6, 7, 8]
```

![merge_sort-sort](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/merge_sort-sort.png)



## Complexity Analysis

### Time Complexity

Best/Average/Worst Case Complexity: $O(n \log n)$

### Space Complexity

The space complexity of merge sort is $O(n)$.

## Reference

#### Mergesort algorithm 👍

{{< youtube TzeBrDU-JaY>}}

#### [Merge Sort Algorithm 👍](https://www.programiz.com/dsa/merge-sort)

