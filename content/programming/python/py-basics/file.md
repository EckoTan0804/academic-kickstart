---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 18

# Basic metadata
title: "Working with Files"
date: 2020-11-19
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
        weight: 8
---

## `with open(...) as ...` pattern

1. `open()` opens file for reading or writing and returns a file handle
2. Use appropriate methods to read or write this file handel

Example

- Read

  ```python
  with open('data.txt', 'r') as f:
      data = f.read()
  ```

- Write

  ```python
  with open('data.txt', 'w') as f:
      data = 'some data to be written to the file'
      f.write(data)
  ```



## Directory listing 

### Directory Listing in Legacy Python Versions

`os.listdir()`

- returns a Python list containing the names of the files and subdirectories in the directory given by the path argument:

Example

```python
import os

entries = os.listdir('my_directory/')
```

### Directory Listing in Modern Python Versions

#### `os.scandir()`

- returns an iterator

  ```python
  ROOT = 'my_directory/'
  entries = os.scandir(ROOT)
  entries
  ```

  ```
  <posix.ScandirIterator at 0x7f6db713c3f0>
  ```

- The `ScandirIterator` points to all the entries in the current directory. You can loop over the contents of the iterator and print out the filenames:

  ```python
  with os.scandir(ROOT) as entries:
      for entry in entries:
          print(f'{entry.name:10}: {entry.path}')
  ```

#### `pathlib` module

- `pathlib.Path()` objects have an `.iterdir()` method for creating an iterator of all files and folders in a directory. Each entry yielded by `.iterdir()` contains information about the file or directory such as its name and file attributes.

- `pathlib` offers a set of classes featuring most of the common operations on paths in an easy, object-oriented way. Another benefit of using pathlib over os is that it reduces the number of imports you need to make to manipulate filesystem paths. :clap:

Example

```python
from pathlib import Path

entries = Path(ROOT)
for entry in entries.iterdir():
    print(f'{entry.name}, name: {entry.stem}, ext: {entry.suffix}')

```

### Summary

| Function                 | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `os.listdir()`           | Returns a list of all files and folders in a directory       |
| `os.scandir()`           | Returns an iterator of all the objects in a directory including file attribute information |
| `pathlib.Path.iterdir()` | Returns an iterator of all the objects in a directory including file attribute information |

### Listing all files in a directory

Filter out directories and only list files from a directory listing

#### Use `os.listdir()`

```python
import os

basepath = 'my_directory'
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        print(entry)
```

#### Use `os.scandir()`

Using `os.scandir()` has the advantage of looking cleaner and being easier to understand than using `os.listdir()`

```python
import os

basepath = 'my_directory'
with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_file():
            print(entry.name)
```

#### Use `pathlib.Path()`

```python
from pathlib import Path

basepath = 'my_directory'
files_in_basepath = Path(basepath).iterdir()
for item in files_in_basepath:
    if item.is_file():
        print(item.name)
```

The code above can be made more concise if we combine the `for` loop and the `if` statement into a single generator expression

```python
from pathlib import Path

basepath = 'my_directory'
files_in_basepath = (entry for entry in Path(basepath).iterdir() if entry.is_file())
for item in files_in_basepath:
    print(item.name)
```

### Listing subdirectories

#### Use `os.listdir()` and

```python
import os

basepath = 'my_directory'
sub_dirs = (entry for entry in os.listdir(basepath) 
            if os.path.isdir(os.path.join(basepath, entry)))

for sub_dir in sub_dirs:
    print(sub_dir)
```

#### Use `os.scandir()`

```python
import os

basepath = 'my_directory'
with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_dir():
            print(entry.name)
```

#### Use `pathlib.Path`

```python
from pathlib import Path

basepath = 'my_directory'
sub_dirs = (item for item in Path(basepath).iterdir() if item.is_dir())
for sub_dir in sub_dirs:
    print(sub_dir.name)
```



## Making directories

`os` and `pathlib` include functions for creating directories. We’ll consider these:

| Function               | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `os.mkdir()`           | Creates a single subdirectory                                |
| `pathlib.Path.mkdir()` | Creates single or multiple directories                       |
| `os.makedirs()`        | Creates multiple directories, including intermediate directories |

### Making single directories

#### Use `os.mkdir()`

To create a single directory, pass a path to the directory as a parameter to `os.mkdir()`:

```python
import os

os.mkdir('example_directory/')
```

Note: If a directory already exists, `os.mkdir()` raises `FileExistsError`. 

#### Use `pathlib`

```python
from pathlib import Path

p = Path('example_directory/')
p.mkdir()
```

If the path already exists, `Path.mkdir()` raises a `FileExistsError`.

To avoid the error, catch the error when it happens and let the user know:

```python
from pathlib import Path

p = Path('example_directory/')
try:    
    p.mkdir()
except FileExistsError as err:
    print(err)
```

Alternatively, you can ignore the `FileExistsError` by passing the `exist_ok=True` argument to `.mkdir()`. This will not raise an error if the directory already exists.

```python
from pathlib import Path

p = Path('example_directory/')
p.mkdir(exist_ok=True)
```

### Creating Multiple Directories

#### Use `os.makedirs`

`os.makedirs()` 

-  similar to `os.mkdir()`
- The difference between the two is that not only can `os.makedirs()` create individual directories, it can also be used to create directory trees. In other words, it can create any necessary intermediate folders in order to ensure a full path exists.

- similar to running `mkdir -p` in Bash.

```python
import os

os.makedirs('dir/sub_dir/sub_sub_dir')
```

This will create a nested directories with default permissions:

```
.
|
└── dir/
    └── sub_dir/
        └── sub_sub_dir/
```

#### Use `pathlib.Path`

```python
import pathlib

p = pathlib.Path('dir/sub_dir/sub_sub_dir')
p.mkdir(parents=True, exist_ok=True)
```

**I prefer using `pathlib` when creating directories because I can use the same function to create single or nested directories.**



## Filename Pattern Matching

| Function                             | Description                                                  |
| ------------------------------------ | ------------------------------------------------------------ |
| `startswith()`                       | Tests if a string starts with a specified pattern and returns `True`or `False` |
| `endswith()`                         | Tests if a string ends with a specified pattern and returns `True` or `False` |
| `fnmatch.fnmatch(filename, pattern)` | Tests whether the filename matches the pattern and returns `True` or `False` |
| `glob.glob()`                        | Returns a list of filenames that match a pattern             |
| `pathlib.Path.glob()`                | Finds patterns in path names and returns a generator object  |





## Traversing Directories and Processing Files

### `os.walk()`

`os.walk()` defaults to traversing directories in a top-down manner.

- To traverse the directory tree in a bottom-up manner, pass in a `topdown=False` keyword argument

`os.walk()` returns three values on each iteration of the loop:

1. The name of the current folder
2. A list of folders in the current folder
3. A list of files in the current folder



## Deleting files

| Function                | Note                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `os.remove()`           | Will throw an `OSError` if the path passed to them points to a directory instead of a file |
| `os.unlink()`           | <li> semantically identical to `os.remove()`                 |
| `pathlib.Path.unlink()` |                                                              |

## Deleting Directories

### Delete single directory

| Function          | Note                                                         |
| ----------------- | ------------------------------------------------------------ |
| `os.rmdir()`      | Only work if the directory you’re trying to delete is empty. If the directory isn’t empty, an `OSError` is raised. |
| `pathlib.rmdir()` | <li> semantically identical to `os.rmdir()`                  |

### Delete entire directory trees

`shutil.rmtree(dir)`: Everything in `dir` is deleted when `shutil.rmtree()` is called on it.



## Summary for deleting

| Function                | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `os.remove()`           | Deletes a file and does not delete directories               |
| `os.unlink()`           | Is identical to `os.remove()` and deletes a single file      |
| `pathlib.Path.unlink()` | Deletes a file and cannot delete directories                 |
| `os.rmdir()`            | Deletes an empty directory                                   |
| `pathlib.Path.rmdir()`  | Deletes an empty directory                                   |
| `shutil.rmtree()`       | Deletes entire directory tree and can be used to delete non-empty directories |

## Copying 

### Copying files

| Function         | Note                                                         |
| ---------------- | ------------------------------------------------------------ |
| `shutil.copy()`  | <li> comparable to the `cp` command in UNIX based systems. <li> `shutil.copy(src, dst)` will copy the file `src` to the location specified in `dst`. If `dst`is a file, the contents of that file are replaced with the contents of `src`. If `dst` is a directory, then `src` will be copied into that director  <li> `shutil.copy()` ONLY copies the file’s contents and the file’s permissions. Other metadata like the file’s creation and modification times are not preserved. |
| `shutil.copy2()` | preserve all file metadata when copying                      |

### Copying directories

`shutil.copytree()`

## Moving Files and Directories

To move a file or directory to another location, use `shutil.move(src, dst)`.

- `src`: file or directory to be moved
- `dst`: destination. If `dst` does not exist, `src` will be renamed to `dst`

## Renaming Files and Directories

- `os.rename(src, dst)`
- `pathlib.Path.rename()`



## Colab Notebook

[Colab Notebook](https://colab.research.google.com/drive/1ey4Rj0hYx0RDMqxIfB7S0SYstf0Crh2x?authuser=1#scrollTo=m1v51QdqGDYs)

## Reference

- [Working with Files in Python](https://realpython.com/working-with-files-in-python/#an-easier-way-of-creating-archives)