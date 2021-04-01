---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 207

# Basic metadata
title: "IP Address & Subnet"
date: 2021-04-01
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Telematics", "Understanding"]
categories: ["Telematics"]
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
    telematics:
        parent: TM-understanding
        weight: 7

---

Let's take `10.0.0.0/8` as an example.

The address `10.0.0.0/8` comprises of two parts

- **IP address (`10.0.0.0`**)

  the global addressing scheme used under Internet Protocol

- **Subnet or IP block (`/8`)**
  - divide the IP addresses into small blocks/ranges. 
  - The "/" notation along with the number is called **prefix**.

Calculation of IP range

- The "/ " notation after the IP address can be used to calculate the IP address range that falls under that category.
- All you have to do is **subtract** the prefix from the number 32 (As IP addresses is a 32 bit number). Put the result as an exponent of 2 and you will get the number of IPs in that range.

For example

- to find the IP range of "/8" prefix, we subtract The prefix 8 from 32. The result 24 is used as an exponent of 2. Hence, the IP range you get is "2 to the power 24" i.e 16777216 IPs.
- Thus, "`10.0.0.0/8`" refers to an IP block ranging from "`10.0.0.0`" to "`10.255.255.255`".