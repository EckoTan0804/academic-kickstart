---
# Basic info
title: "Remove Key from Python Dictionary"
linktitle: ""
date: 2020-12-23
draft: false
# type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics", "dict"]
categories: ["Coding"]
toc: true # Show table of contents?

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: "Different methods to remove keys from python dict"
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
---

If you want to remove keys from Python dictionary, there're different ways to do it:

- [Use Python built-in `del`](#built-in-del)
- [Use `dict.pop()`](#dictpop)
- [Use dictionary comprehension to create a new dictionary without the keys that need to be removed](#use-dictionary-comprehension)

Let's say we have a dict like this:

```python
person_info = {
    "name": "Ecko",
    "age": 20,
    "gender": "male"
}
```

and we want to remove the key `age`.





## Built-in `del`

```python
del person_info["age"]
print(person_info)
```

```
{'name': 'Ecko', 'gender': 'male'}
```

However, if the key doesn't exist, `del` will raise an `KeyError` error. For example:

```python
del person_info["address"]
```

```
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-14-8391b7d3a4c8> in <module>()
----> 1 del person_info["address"]

KeyError: 'address'
```

Therefore, we should check existence before removing:

```python
key = "age"
if key in person_dict:
    del key
```





## `dict.pop()`

To delete a key regardless of whether it is in the dictionary, use `dict.pop()`

> [`pop(key[, default])`](https://docs.python.org/3/library/stdtypes.html#dict.pop)
>
> If *key* is in the dictionary, remove it and return its value, else return *default*. If *default* is not given and *key* is not in the dictionary, a [`KeyError`](https://docs.python.org/3/library/exceptions.html#KeyError) is raised.

The advantage of this way is that you can get and delete an entry from a dict in one line of code. :clap:

For example, we remove the existed key `age`:

```python
age = person_info.pop("age")

print(age)
print(person_info)
```

We remove the key `address` which does NOT exist:

```python
addr = person_info.pop("address", "earth")

print(addr)
print(person_info)
```

```
earth
{'name': 'Ecko', 'age': 20, 'gender': 'male'}
```





## Use dictionary comprehension

```python
{key:val for key, val in dict.items() if key != remove_key}
```

For our example:

```python
{key:val for key,val in person_info.items() if key != "age"}

print(person_info)
```

```
{'gender': 'male', 'name': 'Ecko'}
```

Using dictionary comprehension, the a new dict will be created, and the original dict won't be affected.



## Reference

- [How to remove a key from a Python dictionary?](https://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-python-dictionary)