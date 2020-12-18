---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 201

# Basic metadata
title: "Command: export"
date: 2020-12-17
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
        weight: 1
---

## `export` command

- A built-in command of the Bash shell

- Ensure environment variables and functions to be passed to child processes

- Syntax

  ```bash
  export [-f] [-n] [name[=value] ...] or export -p  
  ```

  - `-f` : Names are exported as functions
  - `-n` : Remove names from export list
  - `-p` : List of all names that are exported in the current shell



## Usage

### Display all the exported environment variables of your system

```bash
export
```

### Display all exported variable on current shell

```bash
export -p
```

If you want to check specified variable:

```bash
export -p | grep <var>
```

### Using export with functions

```bash
export -f function_name  
```

Example

![linux-export-command3](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/linux-export-command3.png)

### Assign a value before exporting a function or variable

```bash
export name[=value]  
```

### **Set an environment variable**

To create a new variable, use the export command followed by a variable name and its value.

Syntax:

```bash
export NAME=VALUE  
```

Example:

```bash
export PROJECT_DIR=main/app
export -p | grep PROJECT_DIR
```

```
declare -x PROJECT_DIR="main/app"
```

{{% alert note %}} 

Variables set directly with `export `are temporary variables, which means that the value defined for that variable will not take effect when you exit the current shell.

{{% /alert %}}

## Environment varaible `PATH`

The `PATH `variable is **an environment variable that contains an ordered list of paths** that Unix/Linux will search for executables when running a command. Using these paths means that we do not have to specify an absolute path when running a command.

For example

```bash
echo $PATH
```

```
/usr/lib/lightdm/lightdm:/usr/local/sbin:/usr/local/bin
```

Here `:` is the separator. The `PATH` variable is itself a **list of folders** that are "walked" through when you run a command. In this case, the folders on `PATH` are:

- `/usr/lib/lightdm/lightdm`
- `/usr/local/sbin`
- `/usr/local/bin`

Linux/Unix traverses these folders in order until finding an executable. 

### Adding a New Path

We can add a new path to the PATH variable using the [*export*](https://linux.die.net/man/1/export) command.

To append a new path, we reassign PATH with our **new path `<new-path>` at the end**:

```bash
export PATH=$PATH:<new-path>
```

For example, let's say we're gonna add a new path `usr/local/bin` to the`PATH` variable:

```bash
export PATH=$PATH:usr/local/bin
```

## Reference

- [Linux export command](https://www.javatpoint.com/linux-export-command)

- [Linux export 命令](https://www.runoob.com/linux/linux-comm-export.html)

- [What does the colon do in PATH](https://stackoverflow.com/questions/35737627/what-does-the-colon-do-in-path)

- [Adding a Path to the Linux PATH Variable](https://www.baeldung.com/linux/path-variable)