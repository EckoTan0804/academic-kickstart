---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 1001

# Basic metadata
title: "Linked List"
date: 2021-04-15
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Algo", "Basics", "Data Structure", "Linked List", "LeetCode"]
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
        parent: leetcode
        weight: 1
---

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

## Draw It out!

To solve the linked list problems, the most important thing is to draw the linked list and the procedure out. The figure can help you to think and code.

## 2 Main Types of Problems

- Modification of links
- Concatenation of linked lists

## To Be Noticed

### Loops

When modifying links, it is easy to create loop uncarefully. To prevent this problem, **draw the list out**! With the help of figure, we can immediately notice and avoid loops.

### Corner Cases

Common corner cases are

- Linked list is empty
- Linked list contains only one node

## Techniques

### Dummy Head

- Add a `dummy_head` before `head`

  ```python3
  dummy_head = ListNode(val=-1, next=head)
  ```

![截屏2021-04-20 00.56.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-04-20%2000.56.48.png)

- `dummy_head.next` is always the first nodes after all operations
- Advantage of adding a `dummy_head` is that we can treat `head` as a normal node, which can help us to handle the corner cases easily (e.g. deleting `head` node)

- Leetcode problems
  - 25 

### Fast and Slow pointers

- Since linked list does not support random indexing as array, we have to access nodes starting from `head`
- We can use `fast` and `slow` pointers to simulate the random access, e.g.
  - If we want to access the middle node of the linked list
    - `fast` and `slow` pointers start from `head`. 
    - At each step, `slow` moves one step forward, `fast` moves two steps forward
    - When `fast` reaches the end, `slow` reaches the middle
  - If we want to access the n-th node from the end
    - `slow` starts from `head`, `fast` starts from the n-th node
    - At each step, `slow` and `fast` move one step forward
    - When `fast` reaches the end, `slow` reaches n-th node from the end

- Leetcode problems



### Recursion

- Linked list has the nature of recursion. If we master the idea of recursion, the solution will be suprisingly clean and succinct

- To apply recursion, we just need to consider three questions

  - **Base case**: Unter which situation / When should the recursion be ended?
    - In linked list, base cases are empty list or list containing only one node
  - **Return value**: What should be returned to the previous level?
  - **Goal of current level**: What should be done in the current level?

  > Reference: [三道题套路解决递归问题](https://lyl0724.github.io/2020/01/25/1/)

- Pre-order traversal vs. post-orderrecursion

  - In pre-order traversal, we image that the **previous nodes are already processed** (we don't care how they are processed)

  - In post-order traversal, we image that the **nodes behind are already processed** (we don't care how they are processed)

  - Example: [Swap nodes in pairs](https://leetcode.com/problems/swap-nodes-in-pairs)

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-04-23%2022.47.40.png" alt="截屏2021-04-23 22.47.40" style="zoom:67%;" />

- Example: [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs)

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-04-23%2022.51.06.png" alt="截屏2021-04-23 22.51.06" style="zoom:80%;" />

- Leetcode problems

  - [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs)
  - [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

### Sort Nodes Alphabetically When Modifying Links or Concatenating Nodes

When modifying links or concatenating nodes, it's easy to get lost or create loops uncarefully. A small trick to avoid these problems is to 

- Draw out how the list looks like before and after modification
- Mark the order of nodes alphabetically

Example

We want to move node 3 to the front of node 2. We draw the "before and after" out:

![image](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/89404580-5ca4-47b4-b61c-7ba26cf586f3_1618868088.0298142.png)

Then we mark the order of nodes alphabetically

![image](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/3f51fe85-c465-4656-a9fc-5eed981d3e33_1618868258.3117332.png)