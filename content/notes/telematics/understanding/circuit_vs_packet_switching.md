---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 202

# Basic metadata
title: "Circuit Switching Vs. Packet Switching"
date: 2021-03-11
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
        weight: 2

---

## TL;DR

| Switching Network | Characteristics                                              | Suitable for                                                 | Use Cases                   |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------- |
| Circuit Switching | A dedicated channel or circuit is established for the duration of communications | Communications which require data to be transmitted in real time | Traditional telephone calls |
| Packet Switching  | Connected through many routers, each serving different segment of network | More flexible and more efficient if some amount of delay is acceptable | Handles digital data        |



## Circuit Switching

- **A dedicated channel or circuit is established for the duration of communications.**

- The method used by the old traditional telephone call, carried over the **Public Switched Telephone Network (PSTN)**
- Also referred to as the **Plain Old Telephone Service (POTS)**
- Ideal for communications which require data to be transmitted in real time

- Normally used for traditional telephone calls

This is what a typical traditional telephone network look like. 

![截屏2021-03-11 17.10.10](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2017.10.10.png)

The PSTN networks are connected through central offices, which act as telephone exchanges, each serving a certain geographical area.

When person A calls Person B:

![Circuit_switching](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Circuit_switching.gif)



## Packet Switching

- Packet switching networks are connected through many routers, each serving different segment of networks

- How it works?

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_1.gif" alt="Packet_Switching_1" style="zoom:67%;" />

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_2.gif" alt="Packet_Switching_2" style="zoom:67%;" />

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_3.gif" alt="Packet_Switching_3" style="zoom:67%;" />

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_4.gif" alt="Packet_Switching_4" style="zoom:67%;" />

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_5.gif" alt="Packet_Switching_5" style="zoom:67%;" />

- More flexible and more efficient if some amount of delay is acceptable

- Normally handle digital data

## Reference

Circuit Switching vs. Packet Switching

{{< youtube B1tElYnFqL8>}}