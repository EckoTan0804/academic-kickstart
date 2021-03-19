---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 206

# Basic metadata
title: "Ethernet Basics"
date: 2021-03-15
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
        weight: 6

---



## CSMA/CD

**CSMA/CD** = **C**arrier **S**ense **M**ultiple **A**ccess with **C**ollision **D**etection

- Media access control method used in early Ethernet technology

**C**arrier **S**ense **M**ultiple **A**ccess

- **Carrier**: transmission medium that carries data, e.g.

  - Electronic bus in Ethernet network

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2010.28.05.png" alt="截屏2021-03-18 10.28.05" style="zoom:67%;" />

  - Band of the electronmagnetic spectrum (channel) in Wi-Fi network

    ![截屏2021-03-18 10.28.43](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2010.28.43.png)

- **Carrier Sense**
  
- A node (i.e. Network Interface Card, NIC) on a network has a sense: it can listen and hear. It can detect what is going on over the transmission medium.
  
- **Multiple Access**
  
- Every node in the network has equal right to access to and use the shared medium, but they must take turns 
  
- Putting them together, CSMA means Before a node transmits data, it checks or listens to the medium
  - Medium not busy ➡️ the node sends its data
  - When the node detects the medium is used ➡️ It will back off and wait for a random amount of time and try again

**C**ollision **D**etection: A node can hear collision if it happens

- Example

  Both A and C want to transmit their data. They check the media and find it is not busy. So they send their message at the same time. Collision occurs. When these two nodes hear the collision, they will back off and use some kind of randomization to decide which would go first in order to avoid collision.

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/CSMA_CD.gif" alt="CSMA_CD" style="zoom:80%;" />

## Ethernet Frame

**Frame = a protocol data unit (PDU)**

PDU in different layer of the OSI model is named differently.

![截屏2021-03-18 10.49.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2010.49.23.png)

Among the Ethernet family, frames can be different. For any two devices to communicate, they must have the same type of frames.

 An Ethernet frames has seven main parts:

![截屏2021-03-18 10.51.21](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2010.51.21.png)

- **Preamble**

  ![截屏2021-03-18 11.01.47](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.01.47.png)

  A 64 bit header information telling the receiving node that a frame is coming and where the frame starts

- **Recipient MAC**

  Recipient's MAC address

- **Sender MAc**

  Sender's MAC address

- **Type**

  Tells the recipient the basic type of data, such as IPv4 or IPv6

- **Data**

  - Payload carried by frame, such as IP packet from Network layer.
  - Limit is 1500 Bytes

- **Pad**

  Extra bits to make a frame at least bigger than 64 Bytes (Any data unit < 64 Bytes would be considered as collisions)

- **FCS** = Frame Check Sequence

  Used for error checking and the integrity verfication of a frame

## Spanning Tree Protocol (STP)

### Complete Graph and Spanning Tree

**Complete Graph**

- A graph in which each pair of graph vertices is connected by a line

  - I.e., when all the points are connected by the **maximum** number of lines, we get a complete graph

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.11.43.png" alt="截屏2021-03-18 11.11.43" style="zoom:67%;" />

- In networking field, a complete graph is like a fully meshed network



**Spanning tree**

- All points are connected by a **minimum** number of lines

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.13.12.png" alt="截屏2021-03-18 11.13.12" style="zoom:67%;" />



- From the complete graph above, we can get three spanning trees:

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.15.51.png" alt="截屏2021-03-18 11.15.51" style="zoom:67%;" />

  All three points are connected and no loop is formed.

- Basic features
  - NO loop
  - Minimumly connected (i.e., removing one line will leave some point disconnected)

### Spanning Tree Protocol

Spanning Tree Protocol (STP) 

- **Layer 2 protocol** that runs on bridges and switches and builds a loop-free logical topology.

- 🎯 Main purpose: **eliminate loops**

- Three basic steps:
  1. Select one switch as root bridge (central point on the network)
  2. Choose the shortest path (the least cost) from a switch to the root bridge
     - Path cost is calculated based on link bandwidth: the higher bandwidth, the lower the path cost.
  3. Block links that cause loops while maintaining these links as backups 

<details>
<summary><b>Example</b></summary>

Suppose we have a simple network

![截屏2021-03-18 11.27.53](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.27.53.png)

First, STP elects the root bridge. The lowest bridge ID (priority: MAC address) determins the root bridge. Here swithc A is elected as the root bridge.

![截屏2021-03-18 11.29.12](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.29.12.png)

Next each of other switches chooses the path to the root bridge with least path cost. Here we just skip the details of calculation and mark the path cost for each link.

![截屏2021-03-18 11.31.34](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.31.34.png)

Now let's take a look at switch B. For switch B, there're two paths to reach root bridge, switch A:

- BDCA: costs 7 (2 + 4 + 1)
- BA: costs 2

Therefore, the link BA is chosen as the path from switch B to root bridge A.

![截屏2021-03-18 11.34.57](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.34.57.png)

- RP: Root port, the port with the least cost path to the root bridge
- DP: designated port.		

For switch C and D, the procedure is similar.		![截屏2021-03-18 11.35.59](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.35.59.png)

Note

- A non-root switch can have many designated ports, but it can have only ONE root port.

- All ports of the root bridge are designated ports. On the root bridge, there is NO root port.

Now every switch has found the best path to reach the root bridge. And the links between D and C should be blocked in order to eliminate a loop.

![截屏2021-03-18 11.43.21](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.43.21.png)

Let's look at the blocked link DC

![截屏2021-03-18 11.43.35](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2011.43.35.png)

The port with the lowest switch ID would be selected as the designated port. The other end is blocking port. The blocking port can still receive frames, but it will not forward or send frames. It simply drops them.

</details>

### How STP Elects Root Bridge?

Root bridge election is based on a 8 byte **switch Bridge ID (BID)**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2016.35.46.png" alt="截屏2021-03-18 16.35.46" style="zoom:67%;" />

- 2 Bytes **Priority Field**
- 6 Bytes **Switch MAC address**

Root bridge election process is simple: **All interconnected switches exchange their BIDs**

- Whoever has the **lowest priority field value** would become the root bridge
- If priority filed is equal, whoever has the **lowest MAC address** would become the root bridge

#### BPDU

Every switch multicasts its message, **Hello BPDU**, in which each swich declares ifself the root bridge.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/BPDU.gif" alt="BPDU" style="zoom:67%;" />

**B**ridge **P**rotocol **D**ata **U**nit (**BPDU**) is a frame containing information about spanning ree protocol. 

**Hello BPDU** is used by switches or bridges to share information about themselves. It is used for

- electing a root bridge
- determining ports roles and states
- blocking unwanted links

In other words, Hello BPDU is used to configure a loop-free network.

**Structure of BPDU:**

![截屏2021-03-18 16.52.57](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2016.52.57.png)

Three important fields:

- Root ID: Root Bridge BID
- Root Path cost: The best path cost to the root bridges
- Bridge ID: BPDU sender's ID (Source BID)

 #### The Election Process

One thing need to be kept in mind: Each port of a switch is uniquely identified.

Consider the example above:

**Switch A, B, and C send out their Hello BPDUs. Basically each switch declares itself the root bridge.**

![截屏2021-03-18 17.04.04](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2017.04.04.png)

Let's take a look at Switch A first.

Switch A sends out its Hello BPDU to B and C

- Switch A sets it Root ID to its own BID (*"Hello everyone, I am the root bridge"*)
- Path cost value is set to 0

![截屏2021-03-18 17.05.08](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2017.05.08.png)

Switch B and C do the same thing.

![截屏2021-03-18 17.07.52](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2017.07.52.png)

![截屏2021-03-18 17.08.26](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2017.08.26.png)

Basically they all claim they are the root bridge ("the boss") in their Hello BPDUs.

The problem is: ONLY one can be the root bridge.

What they do next is to **compare their Hello BPDUs and to elect a real boss.**

- When Switch A receives Hello BPDUs from B and C, it checks and discards their BPDUs because its bridge ID is lower than B's and C's. So A keeps its original Hello BPDU and still believes it is the root bridge.

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/BPDA_Example_A.gif" alt="BPDA_Example_A" style="zoom:80%;" />

- When Switch B receives the Hello BPDU from C, it compares and finds its Bridge ID is lower (i.e. B's BPDU is superior) thus discards C's Hello BPDU

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/BPDU_Example_BC.gif" alt="BPDU_Example_BC" style="zoom:80%;" />

  When B receives A's Hello BPDU, it finds A's Bridge ID is lower. It would say "Well, Switch A is the winner". 

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/BPDU_Example_BA.gif" alt="BPDU_Example_BA" style="zoom:80%;" />

  Therefore, it modifies its Root ID value by replacing its own bridge ID with Switch A's bridge Id. It also calculates the path cost to switch A (let's say 4), and then sends the modified Hello BPDU to others.

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/BPDU_Example_BA2.gif" alt="BPDU_Example_BA2" style="zoom:80 %;" />

- When Switch C receives Hello BPDU from A and B, C finds A's is a superior BPDU. So C changes the value of the root ID to switch A's Bridge ID. And it calculates the path cost to switch A (let's say 1). Then it sends its modified Hello BPDU to others.

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/BPDU_Example_CA.gif" alt="BPDU_Example_CA" style="zoom:80%;" />

This way, A, B, and C exchange their BPDUs again and agree that the root bridge should be switch A. 

Once the root bridge is decided, path cost to the root bridges are calculated. Root ports, designated ports, and blocked ports are determined.  STP has created a loop-free network! 👏 

![截屏2021-03-18 17.38.51](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2017.38.51.png)

 

## Reference

#### CSMA/CD

{{< youtube K_8KJRhOWIA>}}

#### 7 Part of an Ethernet Frame

{{< youtube qXtS1o1HGso>}}

#### Spanning Tree Protocol (IEEE 802 1D)

{{< youtube Ilpmn-H8UgE>}}

{{< youtube BkGEwrzIK4g>}}