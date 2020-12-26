---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 23

# Basic metadata
title: "glob"
date: 2020-12-25
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics", "File"]
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
        weight: 13
---

The [`glob`](https://docs.python.org/3/library/glob.html#module-glob) module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell. If you need a list of filenames that all have a certain extension, prefix, or any common string in the middle, use [`glob`](https://pymotw.com/2/glob/#module-glob) instead of writing code to scan the directory contents yourself.

## Wildcards

- `*` : matches **zero or more** characters in a segment of a name
- `?` : matches any **single** character in that position in the name
- `[] `: matches characters in the given range
  - `[1-9]` : matches any single digit
  - `[a-z` : mathces any single letter

## Functions

### `glob.glob()`

```python
glob(file_pattern, recursive = False)
```

It retrieves the list of files matching the specified pattern in the `file_pattern `parameter.

- The `file_pattern ` can be an absolute or relative path. It may also contain wild cards such as `*` or `?` symbols.

- The `recursive `parameter is turn off (`False`) by default. When `True`, it recursively searches files under all subdirectories of the current directory.

For example, we have file structure like this:

```
- src
|- code
    |- main.py
    |- test0.py
    |- test1.py
    |- test2.py
    |- test-env.py
    |- test-prod.py
|- text
    |- file_a.txt
    |- file_b.txt
    |- file_c.txt
    |- file_d.txt
|- demo.py
```

We want to get all `.py` files:

```python
import glob

for py_file in glob.glob("src/**/*.py", recursive=True):
    print(py_file) 
```

```
src/demo.py
src/code/test1.py
src/code/test-env.py
src/code/main.py
src/code/test2.py
src/code/test0.py
```

We want to get `test0.py`, `test1.py`, `test2.py`:

```python
import glob

for py_file in glob.glob("src/**/test?.py"):
    print(py_file)
```

```
src/code/test1.py
src/code/test2.py
src/code/test0.py
```

We want to get `file_a.txt`, `file_b.txt`, `file_c.txt`:

```python
import glob

for txt_file in glob.glob("src/**/file_[a-c].txt"):
    print(txt_file)
```

```
src/text/file_a.txt
src/text/file_c.txt
src/text/file_b.txt
```

### `glob.iglob()`

Return an [iterator](https://docs.python.org/3/glossary.html#term-iterator) which yields the same values as [`glob()`](https://docs.python.org/3/library/glob.html#module-glob) without actually storing them all simultaneously (good for large directories).

Example:

```python
import glob

for py_file in glob.iglob("src/**/test?.py"):
    print(py_file)
```

```
src/code/test1.py
src/code/test2.py
src/code/test0.py
```

### `glob.escape()`

Escape all special characters (`'?'`, `'*'` and `'['`). "Escape" means treating special characters as normal character instead of wildcards.

This is useful if you want to match an arbitrary literal string that may have special characters in it.

For example, if we want to get `test-env.py` and `test-prod.py` (both contain special character `-` in the filename):

```python
import glob

for py_file in glob.glob("src/code/*" + glob.escape("-") + "*.py"):
    print(py_file)
```

## Another ways for filename pattern matching

### `fnmatch.fnmatch()` [^1]

```python
fnmatch.fnmatch(filename, pattern)
```

Test whether the `filename` string matches the `pattern `string.

For example, we want to get `test0.py`, `test1.py`, `test2.py`:

```python
import os
import fnmatch

for py_file in os.listdir("src/code"):
    if fnmatch.fnmatch(py_file, "test?.py"):
        print(py_file)
```

### `pathlib.Path.glob()` [^2]

```python
pathlib.Path.glob(pattern)
```

Glob the given relative *pattern* in the directory represented by this path, yielding all matching files (of any kind).

For example, if we want to list all python file:

```python
from pathlib import Path

path = Path("src")

for py_file in path.glob("**/*.py"):
    print(py_file)
```

```
src/demo.py
src/code/test1.py
src/code/test-prod.py
src/code/test-env.py
src/code/main.py
src/code/test2.py
src/code/test0.py
```

{{% alert note %}} 

The “`**`” pattern means “this directory and all subdirectories, recursively”. In other words, it enables recursive globbing. However, using the “`**`” pattern in large directory trees may consume an inordinate amount of time.

{{% /alert%}}

`Path.glob()` is similar to `os.glob()` discussed above. As you can see, `pathlib` combines many of the best features of the `os`, `os.path`, and `glob` modules into one single module, which makes it a joy to use.

#### `pathlib.Path.rglob()` [^3]

This is like calling [`Path.glob()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob) with “`**/`” added in front of the given relative *pattern*.

E.g. list all python file:

```python
for py_file in Path("src").rglob("*.py"):
    print(py_file)
```

```
src/demo.py
src/code/test1.py
src/code/test-prod.py
src/code/test-env.py
src/code/main.py
src/code/test2.py
src/code/test0.py
```



## Summary

| Function                             | Description                                                  |
| ------------------------------------ | ------------------------------------------------------------ |
| `fnmatch.fnmatch(filename, pattern)` | Tests whether the filename matches the pattern and returns `True` or `False` |
| `glob.glob()`                        | Returns a list of filenames that match a pattern             |
| `glob.iglob()`                       | Returns an iterator of filenames that match a pattern        |
| `pathlib.Path.glob()`                | Finds patterns in path names and returns a generator object  |
| `pathlib.Path.rglob()`               | Finds patterns in path names recursively and returns a generator object |

## Reference

- [`glob` documentation](https://docs.python.org/3/library/glob.html)
- [Python: glob匹配文件](https://cloud.tencent.com/developer/article/1150612)

- [Python Glob Module – Glob() Method](https://www.techbeamers.com/python-glob/)
- [Filename Pattern Matching](https://realpython.com/working-with-files-in-python/#filename-pattern-matching)



[^1]: [`fnmatch`](https://docs.python.org/3/library/fnmatch.html#module-fnmatch)
[^2]: https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob

[^3]: https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob