---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 24

# Basic metadata
title: "pathlib"
date: 2020-12-26
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics", "File", "pathlib"]
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
        weight: 14
---

## Why `patblib`?

The `pathlib` module, introduced in Python 3.4 ([PEP 428](https://www.python.org/dev/peps/pep-0428/)), gathers the necessary functionality in one place and makes it available through methods and properties on an easy-to-use `Path` object.

```python
from pathlib import Path
```



## Createing paths

- Use class methods:

  - `Path.cwd()`: get Current Working Directory
  - `Path.home()`: get home directory

- Create from string representation explicitly

  E.g.

  ```python
  Path("/content/src/demo.py")
  ```

- Construct path by joining parts of the path

  E.g.

  ```python
  Path.joinpath(Path.cwd(), "src", "demo.py")
  ```

## Reading and writing files

The built-in `open()` function can use `Path` objects directly. E.g.:

```python
with open(path, mode="r") as f:
	pass
```

For simple reading and writing of files, there are a couple of convenience methods in the `pathlib` library:

- `.read_text()`: open the path in text mode and return the contents as a string.
- `.read_bytes()`: open the path in binary/bytes mode and return the contents as a bytestring.
- `.write_text()`: open the path and write string data to it.
- `.write_bytes()`: open the path in binary/bytes mode and write data to it.

Each of these methods handles the opening and closing of the file, making them trivial to use.

## Components of path

The different parts of a path are conveniently available as properties. Basic examples include:

- `.name`: the file name without any directory
- `.parent`: the directory containing the file, or the parent directory if path is a directory
- `.stem`: the file name without the suffix
- `.suffix`: the file extension
- `.anchor`: the part of the path before the directories

```python
path = Path.joinpath(Path.cwd(), "src", "hello-world.txt")

print(f"Full path: {path}")
print(f"Name: {path.name}")
print(f"Parent: {path.parent}")
print(f"Stem: {path.stem}")
print(f"Suffix: {path.suffix}")
print(f"Anchor: {path.anchor}")
```

```
Full path: /content/src/hello-world.txt
Name: hello-world.txt
Parent: /content/src
Stem: hello-world
Suffix: .txt
Anchor: /
```

## Renaming files

When you are renaming files, useful methods might be `.with_name()` and `.with_suffix()`. They both return the original path but with the name or the suffix replaced, respectively.

## Deleting files

Directories and files can be deleted using `.rmdir()` and `.unlink()` respectively.

## Listing files

- `Path.iterdir()` : iterates over all files in the given directory
- More flexible file listing
  - `Path.glob()`
  - `Path.rglob()` (recursive glob)

## Moving files

```python
Path(current_location).rename(new_location)
```

## Correspondence to tools in the [`os`](https://docs.python.org/3/library/os.html#module-os) module

| os and os.path                                               | pathlib                                                      |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [`os.path.abspath()`](https://docs.python.org/3/library/os.path.html#os.path.abspath) | [`Path.resolve()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve) |
| [`os.chmod()`](https://docs.python.org/3/library/os.html#os.chmod) | [`Path.chmod()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.chmod) |
| [`os.mkdir()`](https://docs.python.org/3/library/os.html#os.mkdir) | [`Path.mkdir()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir) |
| [`os.makedirs()`](https://docs.python.org/3/library/os.html#os.makedirs) | [`Path.mkdir(parents=True)`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir) |
| [`os.rename()`](https://docs.python.org/3/library/os.html#os.rename) | [`Path.rename()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.rename) |
| [`os.replace()`](https://docs.python.org/3/library/os.html#os.replace) | [`Path.replace()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace) |
| [`os.rmdir()`](https://docs.python.org/3/library/os.html#os.rmdir) | [`Path.rmdir()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.rmdir) |
| [`os.remove()`](https://docs.python.org/3/library/os.html#os.remove), [`os.unlink()`](https://docs.python.org/3/library/os.html#os.unlink) | [`Path.unlink()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.unlink) |
| [`os.getcwd()`](https://docs.python.org/3/library/os.html#os.getcwd) | [`Path.cwd()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.cwd) |
| [`os.path.exists()`](https://docs.python.org/3/library/os.path.html#os.path.exists) | [`Path.exists()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.exists) |
| [`os.path.expanduser()`](https://docs.python.org/3/library/os.path.html#os.path.expanduser) | [`Path.expanduser()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.expanduser) and [`Path.home()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.home) |
| [`os.listdir()`](https://docs.python.org/3/library/os.html#os.listdir) | [`Path.iterdir()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.iterdir) |
| [`os.path.isdir()`](https://docs.python.org/3/library/os.path.html#os.path.isdir) | [`Path.is_dir()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.is_dir) |
| [`os.path.isfile()`](https://docs.python.org/3/library/os.path.html#os.path.isfile) | [`Path.is_file()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.is_file) |
| [`os.path.islink()`](https://docs.python.org/3/library/os.path.html#os.path.islink) | [`Path.is_symlink()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.is_symlink) |
| [`os.link()`](https://docs.python.org/3/library/os.html#os.link) | [`Path.link_to()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.link_to) |
| [`os.symlink()`](https://docs.python.org/3/library/os.html#os.symlink) | [`Path.symlink_to()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.symlink_to) |
| [`os.readlink()`](https://docs.python.org/3/library/os.html#os.readlink) | [`Path.readlink()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.readlink) |
| [`os.stat()`](https://docs.python.org/3/library/os.html#os.stat) | [`Path.stat()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.stat), [`Path.owner()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.owner), [`Path.group()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.group) |
| [`os.path.isabs()`](https://docs.python.org/3/library/os.path.html#os.path.isabs) | [`PurePath.is_absolute()`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.is_absolute) |
| [`os.path.join()`](https://docs.python.org/3/library/os.path.html#os.path.join) | [`PurePath.joinpath()`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.joinpath) |
| [`os.path.basename()`](https://docs.python.org/3/library/os.path.html#os.path.basename) | [`PurePath.name`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.name) |
| [`os.path.dirname()`](https://docs.python.org/3/library/os.path.html#os.path.dirname) | [`PurePath.parent`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.parent) |
| [`os.path.samefile()`](https://docs.python.org/3/library/os.path.html#os.path.samefile) | [`Path.samefile()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.samefile) |
| [`os.path.splitext()`](https://docs.python.org/3/library/os.path.html#os.path.splitext) | [`PurePath.suffix`](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.suffix) |

## `pathlib` cheatsheet

[pathlib cheatsheet](https://github.com/chris1610/pbpython/blob/master/extras/Pathlib-Cheatsheet.pdf)

## Google Colab Notebook

Open in [Colab](https://colab.research.google.com/drive/1jKTOzkIFs1ZSyp3xUXugR1C1-5Lv9mPZ#scrollTo=U-MJKEzD1TeG)

## Reference

- Documentation: [`pathlib`](https://docs.python.org/3/library/pathlib.html#module-pathlib) — Object-oriented filesystem paths

- [Python 3's pathlib Module: Taming the File System](https://realpython.com/python-pathlib/#the-problem-with-python-file-path-handling)