---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 16

# Basic metadata
title: "f-string"
date: 2020-10-21
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics"]
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
    python:
        parent: py-basics
        weight: 6
---

Python f-string is the newest Python syntax to do string formatting. Python f-strings provide a faster, more readable, more concise, and less error prone way of formatting strings in Python. :clap:

The f-strings have the `f` prefix and use `{}` brackets to evaluate values.

## Python string formatting

The following example summarizes string formatting options in Python:

```python
name = 'Peter'
age = 23

print('%s is %d years old' % (name, age)) # C style
print('{} is {} years old'.format(name, age)) # python format function
print(f'{name} is {age} years old') # f-string
```

```txt
Peter is 23 years old
Peter is 23 years old
Peter is 23 years old
```

## f-string usage

### f-string expressions

We can put expressions between the `{}` brackets, and the expression will be evaluated inside f-string

```python
bags = 3
apples_in_bag = 12

print(f'There are total of {bags * apples_in_bag} apples')
```

```txt
There are total of 36 apples
```

### f-string dictionaries

```python
user = {'name': 'John Doe', 'occupation': 'gardener'}

print(f"{user['name']} is a {user['occupation']}")
```

```txt
John Doe is a gardener
```

### f-string debug

Python 3.8 introduced the self-documenting expression with the `=` character.

```python
import math

x = 0.8

print(f'{math.cos(x) = }')
print(f'{math.sin(x) = }')
```

```txt
math.cos(x) = 0.6967067093471654
math.sin(x) = 0.7173560908995228
```

### multiline f-string

- The f-strings are placed between round brackets
- Each of the strings is preceded with the `f` character

```python
name = 'John Doe'
age = 32
occupation = 'gardener'

msg = (
    f'Name: {name}\n'
    f'Age: {age}\n'
    f'Occupation: {occupation}'
)

print(msg)
```

```txt
Name: John Doe
Age: 32
Occupation: gardener
```

### f-string calling function

```python
def mymax(x, y):
    return x if x > y else y

a = 3
b = 4

print(f'Max of {a} and {b} is {mymax(a, b)}')
```

```txt
Max of 3 and 4 is 4
```

### f-string objects

Python f-string accepts objects as well; the objects must have either `__str__()` or `__repr__()` magic functions defined.

```python
class User:
    def __init__(self, name, occupation):
        self.name = name
        self.occupation = occupation

    def __repr__(self):
        return f"{self.name} is a {self.occupation}"

u = User('John Doe', 'gardener')

print(f'{u}')
```

```txt
John Doe is a gardener
```

## f-string format

### floats

Floating point values have the `f` suffix. We can also specify the **precision**: the number of decimal places. The precision is a value that goes right after the dot character.

```python
val = 12.3

print(f'{val:.2f}')
print(f'{val:.5f}')
```

```txt
12.30
12.30000
```

### width

The width specifier sets the width of the value. The value may be filled with spaces or other characters if the value is shorter than the specified width.

```python
for x in range(1, 11):
    print(f'{x:02} {x*x:3} {x*x*x:4}')
```

```txt
01   1    1
02   4    8
03   9   27
04  16   64
05  25  125
06  36  216
07  49  343
08  64  512
09  81  729
10 100 1000
```

We can combine floats precision and width, for example:

```python
import numpy as np

a = np.random.rand(20)

for index, value in enumerate(a):
    if index < 10:
        print(f'{index:2}: {value:4.4f}')
    else:
        print(f'{index:2}: {value:4.3f}')
```

```txt
 0: 0.1398
 1: 0.7079
 2: 0.2034
 3: 0.5995
 4: 0.7553
 5: 0.6342
 6: 0.4282
 7: 0.4405
 8: 0.1145
 9: 0.3309
10: 0.127
11: 0.156
12: 0.309
13: 0.845
14: 0.816
15: 0.694
16: 0.291
17: 0.440
18: 0.469
19: 0.305
```

`{value:4.4f}` means

- We will use a width of 4-character to print `value`
- The precision of `value` is 4: we will keep 4 decimal places