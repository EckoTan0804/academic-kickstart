---
# Title, summary, and position in the list
linktitle: "MPLS"
summary: ""
weight: 203

# Basic metadata
title: "Multiprotocol Label Switching (MPLS)"
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
        weight: 3

---

## Recall: Packet Switching and Circuit Switching

Suppose an IP packet is sent from Mumbai, India to Kansas City, Kansas using **Packet switching**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_Example_1.gif" alt="Packet_Switching_Example_1" style="zoom: 50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_Example_2.gif" alt="Packet_Switching_Example_2" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_Example_3.gif" alt="Packet_Switching_Example_3" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_Example_4.gif" alt="Packet_Switching_Example_4" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Packet_Switching_Example_5.gif" alt="Packet_Switching_Example_5" style="zoom:50%;" />

Packet switching is flexible and data path is not fixed. But processing IP information at every router **slows down** transmission.

In contrast, **circuit switching** method is a fixed-path switching method. It is reliable but more expensive.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2018.16.54.png" alt="截屏2021-03-11 18.16.54" style="zoom:67%;" />

## MPLS

MPLS allows IP packets to be forwarded at layer 2 (switching level) without being passed up to layer 3 (routing level).

Let's take a look how MPLS works with the same IP packet sent from Mumbai, India to Kansas City, Kansas.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/MPLS_Example_1.gif" alt="MPLS_Example_1" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/MPLS_Example_2.gif" alt="MPLS_Example_2" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/MPLS_Example_3.gif" alt="MPLS_Example_3" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/MPLS_Example_4.gif" alt="MPLS_Example_4" style="zoom:50%;" />



These routers act like switches on a local network. 

![截屏2021-03-11 18.30.34](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2018.30.34.png)

As a result, MPLS offers potentially faster transmission than traditionally packet switching networks 👏.

**In summary, MPLS can create end-to-end paths that act like circuit-switched connections, but deliver layer 3 IP packet.** 

As we know, routing is the layer 3 function while switching is the layer 2 function. MPLS makes those routers on the Internet act like switches on a local network.

$\rightarrow$ MPLS is also called   **2.5 layer protocol**.



## Reference

MPLS - Multiprotocol Label Switching (2.5 layer protocol)

{{< youtube BuIWNecUAE8>}}

