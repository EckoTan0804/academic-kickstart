---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 150

# Basic metadata
title: "Math"
date: 2020-08-17
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Nerual Network Basics"]
categories: ["Deep Learning"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?

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
    deep-learning:
        parent: nn-basics
        weight: 6

---

- [Softmax and its derivative]({{< relref "softmax.md" >}})