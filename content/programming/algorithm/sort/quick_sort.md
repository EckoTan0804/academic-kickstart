---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 302

# Basic metadata
title: "Quick Sort"
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
        weight: 2
---

## 💡 Idea

1. Pick a pivot
2. Partition the array into two sub-arrays
   - Elements less thant the pivot go to the left sub-array
   - Elements greater thant the pivot go to the right sub-array
3. Call quick sort *recursively* on the two sub-arrays
4. Concat left sub-array, pivot, and right sub-array

## Implementation

Quick sort can be implemented using Divide and Conquer (D&C).

- **Base case** will be 
  - Empty array
  - Array with just one element

  We can just return those arrays as is -- there's nothing to sort.

- **Recursive case**

  Decrease the array to two sub-arrays, apply quick sort on them, until they are decreased to base case.

### Non In-place Implementation

In this version, we choose the first element as pivot.

```python
def quick_sort(array):
    # base case
    if len(array) < 2:
        return array
    else:
        pivot = array[0] # choose the first element as pivot

        # Sub-array of all the elements less than the pivot
        less = [i for i in array[1:] if i <= pivot]

        # Sub-array of all the elements greater than the pivot
        greater = [i for i in array[1:] if i > pivot]

        return quick_sort(less) + [pivot] + quick_sort(greater)
```

## In-place Implementation

To implement in-place quick sort, we won't create any new array. Instead, we just swap the elements.

Here we will choose the last element as our pivot. In order to partition the array, we shift the elements which are less than pivot to the left, and shift the elements which are greater than pivot to the right.

```python
def swap(arr, i, j):
    """
    Swap arr[i] and arr[j]
    """
    arr[i], arr[j] = arr[j], arr[i]

def partition(arr, start, end):
    """
    Partition the array with pivot.
    After partitioning, the array should look like 
    [elements <= pivot, pivot, elements > pivot]
    """
    pivot = arr[end]
    pivot_index = start

    for i in range(start, end):
        print(f"i={i}, arr: {arr}, pivot_index: {pivot_index}")
        if arr[i] < pivot:
            swap(arr, i, pivot_index)
            pivot_index += 1
    swap(arr, pivot_index, end)
    
    return pivot_index
```

```python
arr = [7, 2, 1, 6, 8, 5, 3, 4]
partition(arr, 0, len(arr)-1)
arr
```

```txt
i=0, arr: [7, 2, 1, 6, 8, 5, 3, 4], pivot_index: 0
i=1, arr: [7, 2, 1, 6, 8, 5, 3, 4], pivot_index: 0
i=2, arr: [2, 7, 1, 6, 8, 5, 3, 4], pivot_index: 1
i=3, arr: [2, 1, 7, 6, 8, 5, 3, 4], pivot_index: 2
i=4, arr: [2, 1, 7, 6, 8, 5, 3, 4], pivot_index: 2
i=5, arr: [2, 1, 7, 6, 8, 5, 3, 4], pivot_index: 2
i=6, arr: [2, 1, 7, 6, 8, 5, 3, 4], pivot_index: 2
[2, 1, 3, 4, 8, 5, 7, 6]
```



![quick_sort-partition (1)](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/quick_sort-partition%20%281%29.png)



With the implementation of `partition()`, in-place quick sort is simple:

```python
def quick_sort_inplace(arr, start, end):
    # If start >= end, that means array is empty or contains only one element, which is the base case.
    # We do NOT need to sort the array, just keep it as it is.

    # If start < end, the array contains more than 1 element, which is the recursive case.
    if start < end:
        
        # Partition the array with pivot.
        pivot_index = partition(arr, start, end)

        # Recursively apply quick sort on the left sub-array ([elements <= pivot]) 
        quick_sort_inplace(arr, start, pivot_index-1)

        # Recursively apply quick sort on the right sub-array ([elements > pivot]) 
        quick_sort_inplace(arr, pivot_index+1, end)
```

```python
arr = [7, 2, 1, 6, 8, 5, 3, 4]
quick_sort_inplace(arr, 0, len(arr)-1)
arr
```

```txt
[1, 2, 3, 4, 5, 6, 7, 8]
```



![quick_sort-sort (2)](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/quick_sort-sort%20%282%29.png)



## Complexity Analysis

Quick sort is unique because its performance depends on the pivot choice.

- **Worst case**

  Suppose we always choose the first element as the pivot. And we call quicksort with an array that is *already sorted.* Quicksort doesn’t check to see whether the input array is already sorted. So it will still try to sort it.

  ![截屏2021-04-07 15.50.29](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-04-07%2015.50.29.png)

  Notice that we're NOT splitting the array into two halves. Instead, one of the sub-arrays is always empty. So the call stack is really long. In this case, the stack size is $O(n)$.

- **Best case**

  The array is already sorted, and we always pick the middle element as the pivot.

  ![截屏2021-04-07 15.53.55](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-04-07%2015.53.55.png)

  In this case, the stack size is $O(\log n)$

On every level of the call stack, we "touch" $O(n)$ elements. So each level stack takes $O(n)$ time to complete. 

![截屏2021-04-07 15.57.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-04-07%2015.57.48.png)

In our best case, there're $O(\log n)$ levels. Each level takes $O(n)$ time. Therefore, time of the entire algorithm is
$$
O(n) \cdot O(\log n) = O(n \log n).
$$
In the worst case, there are $O(n)$ levels, so the algorithm will take $O(n) \cdot  O(n) = O(n^2)$ time.

Actually, the best case is also the average case. If we always choose a **random** element in the array as the pivot, quicksort will complete in $ O(n \log n)$ time on average.

## Reference

#### Quicksort algorithm 👍

{{< youtube COk73cpQbFQ>}}

