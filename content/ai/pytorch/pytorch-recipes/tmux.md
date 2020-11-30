---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 906

# Basic metadata
title: "Use tmux"
date: 2020-11-29
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "PyTorch Recipe"]
categories: ["Deep Learning"]
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
    pytorch:
        parent: pytorch-recipes
        weight: 6
---

## What is tmux?

- Tmux is a **terminal multiplexer**

  - You can start a Tmux session and then open multiple **windows** inside that **session**. Each window occupies the entire screen and can be split into rectangular **panes**.

    {{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/concept.jpg" title="tmux sessions, windows, panes" numbered="true" >}}

    

- With Tmux you can easily switch between multiple programs in one terminal, detach them and reattach them to a different terminal.

- Tmux sessions are **persistent**

  - **programs running in Tmux will continue to run even if you get disconnected. (Extremly useful when training neural netwoks on remoter server! :clap:)**

- All commands in Tmux start with a prefix, which by default is `ctrl+b`.

## Installation

### On Ubuntu and Debian

```bash
sudo apt install tmux
```

### On macOS

```bash
brew install tmux
```

## Sessions management

| Command                              | Shortcut      | Functionality                                             |
| ------------------------------------ | ------------- | --------------------------------------------------------- |
| `tmux new -s session_name`           |               | creates a new tmux session named `session_name`           |
| `tmux attach -t session_name`        |               | attaches to an existing tmux session named `session_name` |
| `tmux switch -t session_name`        |               | switches to an existing session named `session_name`      |
| `tmux list-sessions` (`tmux ls`)     |               | lists existing tmux sessions                              |
| `tmux detach `                       | `prefix` +`d` | detach the currently attached session                     |
| `tmux renamesession -t 0 <new-name>` | `prefix` +`$` | rename session 0 to `<new-name>`                          |

## Windows management

| Command                               | Shortcut         | Functionality                      |
| ------------------------------------- | ---------------- | ---------------------------------- |
| `tmux new-window`                     | `prefix` + `c`   | create a new window                |
| `tmux select-window -t :0-9`          | `prefix` + `0-9` | move to the window based on index  |
| `tmux select-window -t <window-name>` |                  | move to the window `<window-name>` |
| `tmux rename-window <new-name>`       | `prefix` + `,`   | rename the current window          |

## Panes management

| Command                | Shortcut       | Functionality                     |
| ---------------------- | -------------- | --------------------------------- |
| `tmux split-window`    | `prefix` + `"` | create a new window               |
| `tmux split-window -h` | `prefix` + `%` | move to the window based on index |
|                        | `prefix` + `x` | kill current pane                 |

## Cheatsheet

[Tmux Cheat Sheet & Quick Reference](https://tmuxcheatsheet.com/)

##  Reference

- [A tmux Crash Course](https://thoughtbot.com/blog/a-tmux-crash-course)
- [Getting started with Tmux](https://linuxize.com/post/getting-started-with-tmux/)

- [Linux下的神器介绍之Tmux分屏器](https://www.jianshu.com/p/6699d9f2685d)
- [Tmux 使用教程](http://www.ruanyifeng.com/blog/2019/10/tmux.html)

- Complete tmux Tutorial

  {{< youtube Yl7NFenTgIo >}}