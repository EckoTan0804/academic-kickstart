---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 203

# Basic metadata
title: "Command: chmod"
date: 2021-01-05
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
        weight: 3
---

## Check permissions

In terminal: `ls –l [file_name]`

For example:

```bash
$ ls -l
```

```
-rw-r--r--  1 EckoTan  staff  12 Jan  5 18:02 hello-world.txt
```

- `-rw-r--r--` : file permission
- `EckoTan` : owner/creator of the file
- `staff` : the group to which that owner belongs to 
- `Jan  5 18:02` : date of creation
- `hello-world.txt` : the file

## Permissions

The permission settings, grouped in a string of characters (`-`, `r`, `w`, `x`), are classified into four sections:

1. **File type**. There are three possibilities for the type. It can either be a regular file (**–**), a directory (**d**) or a link (**i**).
2. **File permission of the user (owner)**
3. **File permission of the owner’s group**
4. **File permission of other users**

![file-permission-syntax-explained](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/file-permission-syntax-explained.jpg)

In permission of user, group, and others, the characters **-**, **r**, **w**, and **x** stand for **None**, **read**, **write**, and **execute**.

- `-`: permission is NOT granted
- `r`: **Read** permissions. The file can be opened, and its content viewed. But it can not be modified (nor added/removed)
- `w`: **Write** permissions. The file can be edited, modified, and deleted.
- `x`: **Execute** permissions. If the file is a script or a program, it can be run (executed).This option is mainly used for running scripts.

### Example

```
-rw-r--r--  1 EckoTan  staff  12 Jan  5 18:02 hello-world.txt
```

`hello-world.txt` is a regular file with read and write permission assigned to the owner, but gives read-only access to the group and others



## Change permissions

The command that executes such tasks is the **`chmod`** command.

Syntax:

```bash
$ chmod [permission] [file_name]
```

There are two ways to define permission:

1. using **symbols** (alphanumerical characters)
2. using the **octal notation method**

### Define File Permission with Symbolic Mode

To use `chmod` to set permissions, we need to tell it:

- **Who**: Who we are setting permissions for?

  - `u` : User, meaning the owner of the file.
  - `g` : Group, meaning members of the group the file belongs to.
  - `o` : Others, meaning people not governed by the `u` and `g` permissions.
  - `a` : All, meaning all of the above.

  (If none of these are used, `chmod` behaves as if “`a`” had been used.)

- **What**: What change are we making? Are we adding or removing the permission?

  - `–` : **Removes** the permission.
  - `+` : Grants the permission. The permission is **added** to the existing permissions. 
  - `=` : **Set** a permission and remove others.

- **Which**: Which of the permissions are we setting?

  - `r`
  - `w`
  - `x`

#### Example

```bash
$ ls -l
```

```
-rw-r--r--  1 EckoTan  staff  12 Jan  5 18:02 hello-world.txt
```

We want to give group members to have write permission on `hello-world.txt`:

```bash
$ chmod g+w hello-world.txt
```

```bash
$ ls -l
```

```
-rw-rw-r--  1 EckoTan  staff    12B Jan  5 18:02 hello-world.txt
```

We want the permission settings look like this

- `u`: read, write, execute
- `g`: read, write
- `o`: read, write

```bash
$ chmod u=rwx,g=rw,o=rw hello-world.txt
```

```bash
$ ls -l
```

```
-rwxrw-rw-  1 EckoTan  staff    12B Jan  5 18:02 hello-world.txt
```

### Define permission using numerical shorthand

Another way to specify permission is by using the **octal/numeric** format. This option is faster, as it requires less typing, although it is not as straightforward as the previous method.

The following example shows how it works:

![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/rwx-standard-unix-permission-bits.png)

Using this method, each of the three permissions (`r`, `w`, `x`) is represented by one of the bits in the binary equivalent of the decimal number.

In this example, 

- The permission of user/owner is `rwx`. This could be represented as `111`, which equals 7 in decimal.
- The permission of group is `r-x`. This could be represented as `101`, which equals 5 in decimal.
- The permission of others is `r--`. This could be represented as `1010, which equals 4 in decimal.

Therefore, to achieve the same permission settings as 

```bash
$ chmod u=rwx,g=rx,o=r hello-world.txt 
```

We can also type:

```bash
$ chmod 754 hello-world.txt
```

```
-rwxr-xr--  1 EckoTan  staff    12B Jan  5 18:02 hello-world.txt
```



## Reference

- [How to Use the chmod Command on Linux](https://www.howtogeek.com/437958/how-to-use-the-chmod-command-on-linux/)

- [Linux File Permission Tutorial: How To Check And Change Permissions](https://phoenixnap.com/kb/linux-file-permissions)

  

