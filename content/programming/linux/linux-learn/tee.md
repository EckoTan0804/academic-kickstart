---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 202

# Basic metadata
title: "Command: tee"
date: 2020-12-18
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Linux", "OS"]
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
    linux:
        parent: linux-learn
        weight: 2
---

**The `tee` command reads from the standard input and writes to both standard output and one or more files at the same time.** 

![How to Use the Tee Command in Linux - Make Tech Easier](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/tee-featured-400x200.png)

`tee` is mostly used in combination with other commands through piping.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/tee-pipe-20201218182022663.png" alt="Introduction to the tee Command - Baeldung on Linux" style="zoom:80%;" />

## Syntax

```sh
tee [OPTIONS] [FILE]
```

Copy

- `OPTIONS`
  - `-a` (`--append`) : Do not overwrite the files instead append to the given files.
  - `-i` (`--ignore-interrupts`) : Ignore interrupt signals.
  - Use `tee --help` to view all available options.
- `FILE`  One or more files. Each of which the output data is written to.

## Usage

### Display output and write it in a file

The most basic usage of the `tee` command is to display the standard output (`stdout`) of a program and write it in a file.

Example:

```bash
$ echo "Hello world!" | tee hello-world.txt
```

```
Hello World!
```

```bash
$ cat hello-world.txt
```

```
Hello World!
```

The output is piped to the `tee` command, which displays the output to the terminal and writes the same information to the file `hello-world.txt`.

### Write to Multiple Files

The `tee` command can also write to multiple files. To do so, specify a list of files separated by space as arguments:

```bash
$ command | tee file1.out file2.out file3.out
```

### Append to File

By default, the `tee` command will overwrite the specified file. Use the `-a` (`--append`) option to [append the output to the file](https://linuxize.com/post/bash-append-to-file/) :

```bash
$ command | tee -a file.out
```

E.g.

```bash
$ echo "Hello world!" | tee hello-world.txt
```

```
Hello world!
```

```bash
$ cat hello-world.txt
```

```
Hello World!
```

```bash
$ echo "Hi world!" | tee -a hello-world.txt
```

```
Hi world!
```

```bash
$ cat hello-world.txt
```

```
Hello World!
Hi world!
```

### Ignore Interrupt

To ignore interrupts use the `-i` (`--ignore-interrupts`) option. This is useful when stopping the command during execution with `CTRL+C` and want `tee` to exit gracefully.

```bash
$ command | tee -i file.out
```

### Hide the Output

If you don’t want `tee` to write to the standard output, you can redirect it to `/dev/null`:

```bash
$ command | tee file.out >/dev/null
```

## Reference

- [Linux Tee Command with Examples](https://linuxize.com/post/linux-tee-command/)
- [Linux tee命令](https://www.runoob.com/linux/linux-comm-tee.html)