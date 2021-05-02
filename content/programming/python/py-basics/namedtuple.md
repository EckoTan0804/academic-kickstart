---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 26

# Basic metadata
title: "Namedtuple"
date: 2020-05-02
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics", ]
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
        weight: 16
---

## TL;DR

- `collection.namedtuple` is a memory-efficient shortcut to defining an immutable class in Python manually.
- Namedtuples can help clean up your code by enforcing an easier to understand structure on your data, which also improves readability.
- Namedtuples provide a few useful helper methods that all start with an `_` underscore—but are part of the public interface.

## What is `namedtuple`?

Python’s tuple is a simple immutable data structure for grouping arbitrary objects. It has two shortcomings:

- The data you store in it can only be pulled out by accessing it through integer indexes. You can’t give names to individual properties stored in a tuple. This can impact code readability.
- It’s hard to ensure that two tuples have the same number of fields and the same properties stored on them. 

`namedtuple` aims to solve these two problems. [namedtuple](http://docs.python.org/2/library/collections.html#collections.namedtuple) is a **factory function** for making a tuple class. With that class we can create tuples that are callable by name also.

- `namedtuple` is immutable just like regular tuple
- Each object stored in them can be accessed through a unique (human-readable) string identifier. This frees you from having to remember integer indexes, or resorting to workarounds like defining integer constants as mnemonics for your indexes. 👏

## How `namedtuple` works?

Let's take a look at an example:

```python
from collections import namedtuple

User = namedtuple("User", ["name", "id", "gender"])
user = User("Ecko", 1, "male")
user
```

```txt
User(name='Ecko', id=1, gender='male')
```

This is equivalent to

```python
class User:
    def __init__(self, name, id, gender):
        self.name = name
        self.id = id
        self.gender = gender

user = User("Ecko", 1, "male")
```

## Operations

### Unpacking

Tuple unpacking and the `*`-operator for function argument unpacking also work as expected:

```python
name, id, gender = user
print(f"name: {name}, id: {id}, gender: {gender}")
```

```txt
name: Ecko, id: 1, gender: male
```

```python
print(*user)
```

```txt
Ecko 1 male
```

### Accessing Values

Values can be accessed either by identifier or by index.

By identifier:

```python
user.name
```

```txt
Ecko
```

By index:

```python
user[0]
```

```ftxt
Ecko
```

### Built-in Helper Functions

- `namedtuple` has some useful helper methods. 

- Their names all start with an underscore character (`_`). 
  - With `namedtuples` the underscore naming convention has a different meaning though: These helper methods and properties are part of namedtuple’s public interface. 
  - The helpers were named that way to avoid naming collisions with user-defined tuple fields.

#### `_asdict`

Returns the contents of a `namedtuple `as a dictionary

```python
user._asdict()
```

```txt
OrderedDict([('name', 'Ecko'), ('id', 1), ('gender', 'male')])
```

We can convert a dictionary into a `namedtuple` with `**` operator

```python
user_dict = {"name": "Ben", "id": 2, "gender": "male"}
User(**user_dict)
```

```txt
User(name='Ben', id=2, gender='male')
```

#### `_replace()`

Creates a (shallow) copy of a tuple and allows you to selectively replace some of its fields.

```python
user._replace(id=2)
```

```txt
User(name='Ecko', id=2, gender='male')
```

#### `_make()`

Classmethod can be used to create new instances of a namedtuple from a sequence or iterable

```python
User._make(["Ilona", 3, "female"])
```

```txt
User(name='Ilona', id=3, gender='female')
```

## When Use `namedtuple`?

- If you're going to create a bunch of instances of a class and NOT change the attributes after you set them in `__init__`, you can consider to use `namedtuple`.
  - Example: Definition of classes in [`torchvision.datasets.cityscapes`](https://pytorch.org/vision/stable/_modules/torchvision/datasets/cityscapes.html)
- Namedtuples can be an easy way to clean up your code and to make it more readable by enforcing a better structure for your data. You should use `namedtuple` **anywhere you think object notation will make your code more pythonic and more easily readable**

## Reference

- [Writing Clean Python With Namedtuples](https://dbader.org/blog/writing-clean-python-with-namedtuples)

- [When and why should I use a namedtuple instead of a dictionary?](https://stackoverflow.com/questions/9872255/when-and-why-should-i-use-a-namedtuple-instead-of-a-dictionary)
- [What are “named tuples” in Python?](https://stackoverflow.com/questions/2970608/what-are-named-tuples-in-python)

- [什么时候使用 namedtuple](https://blog.csdn.net/zV3e189oS5c0tSknrBCL/article/details/78496429)

- [Namedtuple in Python](https://www.geeksforgeeks.org/namedtuple-in-python/)