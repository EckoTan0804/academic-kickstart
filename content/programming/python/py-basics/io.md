---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 19

# Basic metadata
title: "I/O"
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
        weight: 9
---

## What is a File?

At its core, a file is a contiguous set of bytes [used to store data](https://en.wikipedia.org/wiki/Computer_file). This data is organized in a specific format and can be anything as simple as a text file or as complicated as a program executable. In the end, these byte files are then translated into binary `1`and `0` for easier processing by the computer.

Files on most modern file systems are composed of three main parts:

1. **Header:** metadata about the contents of the file (file name, size, type, and so on)
2. **Data:** contents of the file as written by the creator or editor
3. **End of file (EOF):** special character that indicates the end of the file

![The file format with the header on top, data contents in the middle and the footer on the bottom.](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/FileFormat.02335d06829d-20201119181934842-20201119210325141.png)

## Opening and closing files

- Use `with` statement

  ```python
  with open('dog_breeds.txt', 'r') as reader:
      # Further file processing goes here
      pass
  ```
  - The second positional argument, `mode`, is a string that contains multiple characters to represent how you want to open the file. The default and most common is `'r'`, which represents opening the file in read-only mode as a text file

  - Other options for modes are [fully documented online](https://docs.python.org/3/library/functions.html#open), but the most commonly used ones are the following:

    | Character        | Meaning                                                      |
    | ---------------- | ------------------------------------------------------------ |
    | `'r'`            | Open for reading (default)                                   |
    | `'w'`            | Open for writing, truncating (overwriting) the file first    |
    | `'rb'` or `'wb'` | Open in binary mode (read/write using byte data)             |
    | `'a'`            | open for writing, appending to the end of the file if it exists |

## Reading opened files

| Method                                                       | What It Does                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`.read(size=-1)`](https://docs.python.org/3.7/library/io.html#io.RawIOBase.read) | This reads from the file based on the number of `size`bytes. If no argument is passed or `None` or `-1` is passed, then the entire file is read. |
| [`.readline(size=-1)`](https://docs.python.org/3.7/library/io.html#io.IOBase.readline) | This reads at most `size` number of characters from the line. This continues to the end of the line and then wraps back around. If no argument is passed or `None` or `-1` is passed, then the entire line (or rest of the line) is read. |
| [`.readlines()`](https://docs.python.org/3.7/library/io.html#io.IOBase.readlines) | This reads the remaining lines from the file object and returns them as a list. |

Example:

```python
with open('dog_breeds.txt', 'r') as reader:
    for line in reader.readlines():
        print(line, end='') 
```

{{% alert note %}} 

The `end=''` is to prevent Python from adding an additional newline to the text that is being printed and only [print](https://realpython.com/courses/python-print/) what is being read from the file.

{{% /alert %}}

The code above can be further simplified by iterating over the file object itself:

```python
with open('dog_breeds.txt', 'r') as reader:
    for line in reader:
        print(line, end='')
```

This approach is more Pythonic and can be quicker and more memory efficient. Therefore, it is suggested you use this instead.

## Writing opened files

| Method             | What It Does                                                 |
| ------------------ | ------------------------------------------------------------ |
| `.write(string)`   | This writes the string to the file.                          |
| `.writelines(seq)` | This writes the sequence to the file. No line endings are appended to each sequence item. It’s up to you to add the appropriate line ending(s) |

## Tips and tricks

### Working With Two Files at the Same Time

```python
d_path = 'dog_breeds.txt'
d_r_path = 'dog_breeds_reversed.txt'

with open(d_path, 'r') as reader, open(d_r_path, 'w') as writer:
    dog_breeds = reader.readlines()
    writer.writelines(reversed(dog_breeds))
```

### Get rid of `\n` when reading lines

Reference: [How to read a file without newlines?](https://stackoverflow.com/questions/12330522/how-to-read-a-file-without-newlines)

```python
with open(filename) as f:
    mylist = f.read().splitlines() 
```



## Google Colab Notebook

[Colab Notebook](https://colab.research.google.com/drive/1IH4DzZQ6IoNJCDIAZkuJjL7tRqM0Yyfb?authuser=1#scrollTo=8C1Yfj64FvdB)

## Reference

- [Reading and Writing Files in Python (Guide)](https://realpython.com/read-write-files-python/)