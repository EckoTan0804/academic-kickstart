---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 601

# Basic metadata
title: "IPython and Shell Commands"
date: 2020-11-07
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "IPython", ]
categories: ["coding"]
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
        parent: ipython
        weight: 1

---

## Shell Commands in IPython

Any command that works at the command-line can be used in IPython by prefixing it with the `!` character.

Example

```python
! echo 'hello world!' # echo is like Python's print function
```

```txt
hello world!
```

```python
! pwd # = print working dir
```

```txt
/tmp
```

{{% alert note %}} 

For Unix shell, more see: [The Unix Shell](http://swcarpentry.github.io/shell-novice/)

{{% /alert %}}

## Passing Values to and from the Shell

Shell commands can not only be called from IPython, but can also be made to interact with the IPython namespace. For example, we can save the output of any shell command to a Python list using the assignment operator.

For example

```python
contents = !ls
contents
```

```txt
['my-project']
```

```python
directory = !pwd
```

```txt
directory
```

Note that these results are not returned as lists, but as a special shell return type defined in IPython:

```python
type(directory)
```

```txt
IPython.utils.text.SList
```

This looks and acts a lot like a Python list, but has additional functionality, such as the `grep` and `fields` methods and the `s`, `n`, and `p` properties that allow us to search, filter, and display the results in convenient ways.

Communication in the other direction–passing Python variables into the shell–is possible using the `{varname} `syntax:

```python
name = 'Ben'
! echo "hello {name}"
```

```txt
hello Ben
```

## Shell-Related agic Commands

Shell commands in the notebook are executed in a **temporary subshell**. If we'd like to execute the shell commands in a more enduring way, we can use the `%` magic command.

With `%automagic` magic function, we can enable `automagic` function. Then available shell-like magic functions, such as `%cd `, `%cat`, `%cp`, `%env`, `%ls`, `%man`, `%mkdir`, `%more`, `%mv`, `%pwd`, `%rm`, and `%rmdir`, can be used without the `%` sign.

## Magic Commands

There are two types of magic commands

- Line magics: start with `%`
- Cell magics: start with `%%`

Useful magic commands:

- `%matplotlib`: activates matplotlib interactive support during an IPython session
- `%run`: runs a Python script from within IPython shell
- `%time`: displays time required by IPython environment to execute a Python expression.
- `%timeit`: uses the Python [timeit module](https://docs.python.org/3.5/library/timeit.html) which runs a statement 100,000 times (by default) and then provides the mean of the fastest three times.

## Reference

- [IPython and Shell Commands](https://jakevdp.github.io/PythonDataScienceHandbook/01.05-ipython-and-shell-commands.html)

- [IPython - Magic Commands](https://www.tutorialspoint.com/jupyter/ipython_magic_commands.htm)

- [28 Jupyter Notebook Tips, Tricks, and Shortcuts](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)