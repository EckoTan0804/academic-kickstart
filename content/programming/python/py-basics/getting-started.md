---
# Title, summary, and position in the list
linktitle: ""
summary: ""
weight: 10

# Basic metadata
title: "Getting Started"
date: 2020-08-30
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics"]
categories: ["Coding"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?

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
        parent: py-basics
        weight: 1

---


## Source

This is the summary of the book "A Whirlwind Tour of Python" by *Jake VanderPlas*. 

You can view it in 

- nbviewer: [A whirlwind Tour of Python](https://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb), or
- Github: [A whirlwind Tour of Python](https://github.com/jakevdp/WhirlwindTourOfPython)


```python
import this
```

    The Zen of Python, by Tim Peters
    
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!


## Operators

### Identity and Membership

The identity operators, `is` and `is not` check for **object identity**. Object identity is different than equality.


```python
a = [1, 2, 3]
b = [1, 2, 3]
```


```python
a == b
```




    True




```python
a is b
```




    False




```python
1 in a
```




    True




```python
4 in a
```




    False



## Built-in Types

Python's simple types:

|       Type |      Example |                                                  Description |
| ---------: | -----------: | -----------------------------------------------------------: |
|      `int` |      `x = 1` |                               integers (i.e., whole numbers) |
|    `float` |    `x = 1.0` |                  floating-point numbers (i.e., real numbers) |
|  `complex` | `x = 1 + 2j` | Complex numbers (i.e., numbers with real and imaginary part) |
|     `bool` |   `x = True` |                                   Boolean: True/False values |
|      `str` |  `x = 'abc'` |                                   String: characters or text |
| `NoneType` |   `x = None` |                              Special object indicating nulls |

### Complex Numbers


```python
complex(1, 2)
```




    (1+2j)



Alternatively, we can use the "`j`" suffix in expressions to indicate the imaginary part:


```python
1 + 2j
```




    (1+2j)




```python
c = 3 + 4j
```


```python
c.real
```




    3.0




```python
c.imag
```




    4.0




```python
c.conjugate()
```




    (3-4j)




```python
abs(c)
```




    5.0



### String Type


```python
msg = "what do you like?" # double quotes
response = 'spam' # single quotes
```


```python
# length
len(response)
```




    4




```python
# Upper/lower case
response.upper()
```




    'SPAM'




```python
# Capitalize, see also str.title()
msg.capitalize()
```




    'What do you like?'




```python
# concatenation with +
msg + response
```




    'what do you like?spam'




```python
# multiplication is multiple concatenation
5 * response
```




    'spamspamspamspamspam'




```python
# Access individual characters (zero-based (list) indexing)
```


```python
msg[0]
```




    'w'



### None Type

Most commonly used as the default return value of a function


```python
type(None)
```




    NoneType




```python
ret_val = print("abc")
```

    abc



```python
print(ret_val)
```

    None


Likewise, any function in Python with no return value is, in reality, returning `None`.

### Boolean

Booleans can also be constructed using the `bool()` object constructor: values of any other type can be converted to Boolean via predictable rules

-  any numeric type is False if equal to zero, and True otherwise

- The Boolean conversion of `None` is always False

- For strings, `bool(s)` is False for empty strings and True otherwise

- For sequences, the Boolean representation is False for empty sequences and True for any other sequences


```python
# numeric type
bool(0)
```




    False




```python
bool(1)
```




    True




```python
a = 0
if not a:
    print("a")
```

    a



```python
bool(None)
```




    False




```python
bool("")
```




    False




```python
bool("Hello World!")
```




    True




```python
bool([])
```




    False




```python
bool([1])
```




    True




```python
l_1 = [1, 2, 3]
l_2 = []

def is_empty(l):
    if l:
        print("not empty")
        return False
    else:
        print("empty")
        return True
```


```python
is_empty([1, 2, 3])
```

    not empty





    False




```python
is_empty([])
```

    empty





    True



## Built-In Data Structures

| Type Name |                 Example |                           Description |
| --------: | ----------------------: | ------------------------------------: |
|    `list` |             `[1, 2, 3]` |                    Ordered collection |
|   `tuple` |             `(1, 2, 3)` |          Immutable ordered collection |
|    `dict` | `{'a':1, 'b':2, 'c':3}` |         Unordered (key,value) mapping |
|     `set` |             `{1, 2, 3}` | Unordered collection of unique values |

## Defining and Using Functions

### `*args` and `**kwargs`

Write a function in which we don't initially know how many arguments the user will pass. 

- `*args`: 

    - `*` before a variable means "expand this as a sequence"

    - `args` is short for "arguments"
    
- `**kwargs`
  
    - `**` before a variable means "expand this as a dictionary"

    - `kwargs` is short for "keyword arguments"


```python
def catch_all(*args, **kwargs):
    print("args = ", args)
    print("kwargs = ", kwargs)
```


```python
catch_all(1, 2, 3, a=4, b=5)
```

    args =  (1, 2, 3)
    kwargs =  {'a': 4, 'b': 5}



```python
inputs = (1, 2, 3)
keywords = {"one": 1, "two": 2}

catch_all(*inputs, **keywords)
```

    args =  (1, 2, 3)
    kwargs =  {'one': 1, 'two': 2}


## Iterators

### `enumerate`

"Pythonic" way to enumerate the indices and values in a list.


```python
l = [2, 4, 6, 8, 10]
for i, val in enumerate(l):
    print("index: {}, value: {}".format(i, val))
```

    index: 0, value: 2
    index: 1, value: 4
    index: 2, value: 6
    index: 3, value: 8
    index: 4, value: 10


### `zip`

Iterate over multiple lists simultaneously


```python
L = [1, 3, 5, 7, 9]
R = [2, 4, 6, 8, 10]

for l_val, r_val in zip(L, R):
    print("L: {}, R: {}".format(l_val, r_val))
```

    L: 1, R: 2
    L: 3, R: 4
    L: 5, R: 6
    L: 7, R: 8
    L: 9, R: 10



```python
for i, val in enumerate(zip(L, R)):
    print("Index: {}, L: {}, R: {}".format(i, val[0], val[1]))
```

    Index: 0, L: 1, R: 2
    Index: 1, L: 3, R: 4
    Index: 2, L: 5, R: 6
    Index: 3, L: 7, R: 8
    Index: 4, L: 9, R: 10


### `map` and `filter`

`map`: takes a function and applies it to the values in an iterator


```python
func = lambda x: x + 1
l = [1, 2, 3, 4, 5]
```


```python
list(map(func, l))
```




    [2, 3, 4, 5, 6]



`filter`: only passes-through values for which the filter function evaluates to `True`


```python
is_even = lambda x: x % 2 == 0
list(filter(is_even, l))
```




    [2, 4]



### Iterators as function arguments

It turns out that the `*args` syntax works not just with sequences, but with any iterator:


```python
print(*range(5))
```

    0 1 2 3 4



```python
list(range(3))
```




    [0, 1, 2]




```python
print(*map(lambda x: x + 1, range(3)))
```

    1 2 3



```python
L1 = [1, 2, 3, 4]
L2 = ["a", "b", "c", "d"]

z = zip(L1, L2)
print(*z)
```

    (1, 'a') (2, 'b') (3, 'c') (4, 'd')



```python
z = zip(L1, L2)
new_L1, new_L2 = zip(*z)
new_L1
```




    (1, 2, 3, 4)




```python
new_L2
```




    ('a', 'b', 'c', 'd')



### Specialized Iterators: `itertools`


```python
from itertools import permutations

p = permutations(range(3))
print(*p)
```

    (0, 1, 2) (0, 2, 1) (1, 0, 2) (1, 2, 0) (2, 0, 1) (2, 1, 0)



```python
p
```




    <itertools.permutations at 0x10fb32710>



## List Comprehensions


```python
l = [1, 2, 3, 4, 5]
```


```python
[2 * el for el in l if el > 3]
```




    [8, 10]



which is equivalent to the loop syntax, but list comprehension is much easier to write and to understand!


```python
L = []
for el in l:
    if el > 3:
        L.append(2 * el)
        
L
```




    [8, 10]




```python
[(i, j) for i in range(2) for j in range(3)]
```




    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]




```python
print(*range(10))

# Leave out multiples of 3, and negate all multiples of 2
[val if val % 2 else -val for val in range(10) if val % 3]
```

    0 1 2 3 4 5 6 7 8 9





    [1, -2, -4, 5, 7, -8]




```python
L = []

for val in range(10):
    if val % 3 != 0: # conditional on iterator 
        # conditional on value
        if val % 2 != 0:
            L.append(val)
        else:
            L.append(-val)

L
```




    [1, -2, -4, 5, 7, -8]




```python
{n * 2 for n in range(5)}
```




    {0, 2, 4, 6, 8}




```python
{a % 3 for a in range(100)}
```




    {0, 1, 2}



## Generators

Difference between list comprehensions and generator expressions:

### **List comprehensions use square brackets, while generator expressions use parentheses**


```python
# list comprehension:
[n * 2 for n in range(5)]
```




    [0, 2, 4, 6, 8]




```python
# generator
g = (n * 2 for n in range(5))
list(g)
```




    [0, 2, 4, 6, 8]



### **A list is a collection of values, while a generator is a recipe for producing values**

When you create a list, you are actually building a collection of values, and there is some memory cost associated with that. 

When you create a generator, you are not building a collection of values, but a recipe for producing those values. 

Both expose the same iterator interface.


```python
l = [n * 2 for n in range(5)]
for val in l:
    print(val, end=" ")
```

    0 2 4 6 8


```python
g = g = (n * 2 for n in range(5))
for val in g:
    print(val, end=" ")
```

    0 2 4 6 8

The difference is that a generator expression does not actually compute the values until they are needed. This not only leads to memory efficiency, but to computational efficiency as well! This also means that while the size of a list is limited by available memory, the size of a generator expression is unlimited!

### A list can be iterated multiple times; a generator expression is single-use


```python
l = [n * 2 for n in range(5)]

for val in l:
    print(val, end=" ")

print("\n")

for val in l:
    print(val, end=" ")
```

    0 2 4 6 8 
    
    0 2 4 6 8


```python
g = g = (n * 2 for n in range(5))

list(g)
```




    [0, 2, 4, 6, 8]




```python
list(g)
```




    []



This can be very useful because it means iteration can be stopped and started:


```python
g = g = (n ** 2 for n in range(12))

for n in g:
    print(n, end=" ") 
    if n > 30:
        break

print("\nDoing something in between...")

for n in g:
    print(n, end=" ")
```

    0 1 4 9 16 25 36 
    Doing something in between...
    49 64 81 100 121

This is useful when working with collections of data files on disk; it means that you can quite easily analyze them in batches, letting the generator keep track of which ones you have yet to see.

### Generator Functions: Using `yield`


```python
# list comprehension

L1 = [n * 2 for n in range(5)]

L2 = []
for n in range(5):
    L2.append(n * 2)

print("L1:", L1)
print("L2:", L2)
```

    L1: [0, 2, 4, 6, 8]
    L2: [0, 2, 4, 6, 8]



```python
# generator
G1 = (n * 2 for n in range(5))

# generator function
def gen():
    for n in range(5):
        yield n * 2

G2 = gen()

print(*G1)
print(*G2)
```

    0 2 4 6 8
    0 2 4 6 8


### Example: Prime Number Generator


```python
# Generate a list of candidates
L = [n for n in range(2, 40)]
print(L)
```

    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]



```python
# Remove all multiples of the first value
L = [n for n in L if n == L[0] or n % L[0] > 0]
print(L)
```

    [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]



```python
# Remove all multiples of the second value
L = [n for n in L if n == L[1] or n % L[1] > 0]
print(L)
```

    [2, 3, 5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37]


If we repeat this procedure enough times on a large enough list, we can generate as many primes as we wish.

Encapsulate this logic in a generator function:


```python
def gen_primes(N):
    """
    Generate primes up to N
    """
    primes = set()
    for n in range(2, N):
        # print("n = ", n, ":", *(n % p > 0 for p in primes))
        if all(n % p > 0 for p in primes):
            primes.add(n)
            yield n


print(*gen_primes(100))
```

    2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97



```python

```


```python

```