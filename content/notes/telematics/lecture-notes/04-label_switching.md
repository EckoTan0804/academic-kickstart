---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 14

# Basic metadata
title: "Label Switching"
date: 2021-03-10
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Telematics", "Lecture Notes"]
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
        parent: TM-lecture-note
        weight: 4

---

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Label_Switching%20%281%29.png" caption="Summary of Label Switching" numbered="true" >}}

## Motivation

Issues related to IP based routing

- **Lookup is rather complex**
  - Longest matching prefix $\rightarrow$ high performance forwarding needed
- **Shortest path routing selects shortest path to destination**
  - Multiple paths to destination can not be utilized concurrently $\rightarrow$ traffic engineering desirable

- **Strictly packet based**
  - Each IP datagram is handled individually – no support for data streams (flows) 🤪

## Flows

### What is a flow?

A {{< hl >}}flow{{< /hl >}} is a sequence of packets traversing a network that share a set of header field values.

Different levels of granularity possible, e.g.,

- All packets belonging to a particular TCP connection 
- HTTPS traffic
- VoIP traffic
  - Of a particular sender 
  - Within a network

Example

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2013.28.11.png" alt="截屏2021-03-10 13.28.11" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2013.28.35.png" alt="截屏2021-03-10 13.28.35" style="zoom:67%;" />

### Flow Based Forwarding

- Fundamental concept, independent of certain layers 
  - Can span multiple layers
- Incorporates classic routing/forwarding concepts 
- Goes beyond classic concepts

### Aggregation

- **Micro-flows**
  - Consider a single “connection” e.g., a TCP connection
  - Fine grained control

  - High number of flows possible

- **Macro-flows**

  - Higher level of aggregation
  - Aggregation of several “connections”
    - e.g., IP destination address in specific subnet

  - Lower number of flows

## Label Switching

### Classification of Communication Networks

![截屏2021-03-10 13.32.22](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2013.32.22.png)

![截屏2021-03-10 13.32.41](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2013.32.41.png)

### Label Switching

- Combination of

  - **Packet switching**
    - Packets are forwarded individually (data path is NOT fixed)

    - Packets include metadata needed for forwarding decision

  - **Circuit switching**
    - Paths established for flows through the network (data path is fixed)
    - Simple forwarding decision

    - Differentiation of flows possible
      - Load balancing
      - Quality of service (QoS)

- Implementation
  - **Switching** at layer 2, Instead of routing at layer 3
  - **Labels**: Identification which is only locally valid
  - **Virtual circuits**: Sequence of labels

### Label

- Short unstructured identification of fixed length
  - Does NOT carry any layer-3-information

  - Unique: only locally at the corresponding switch 
  - Label swapping: Mapping from input label to output label
- Virtual circuit: Identified through sequence of labels at the path

![截屏2021-03-10 16.55.43](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2016.55.43.png)

### Transport of Label

Label must be transported within the packet

- Additional „header“ in the packet, between headers of layer 2 and layer 3 $\rightarrow$ **layer 2.5**

  ![截屏2021-03-10 17.10.57](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.10.57.png)

- Alternative: In specialized fields within existing packet headers
  
  - IPv6: flow label (20 bit field in IPv6 header, to identify micro flows more easily)

### Label Switching Domain

Basic architecture

- Border of the domain (**edge devices**)
  - Add / remove label

  - Map flow to forwarding class 
  - Access control
  - ...
- Within the domain (**switching device**) 
  - Forward packets based on label information 
  - Label swapping

![截屏2021-03-10 17.12.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.12.48.png)

### Label Forwarding Information Base

Forwarding table in case of label switching: Efficient access through label (NO longest prefix matching needed).

Example: 

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.14.25.png" alt="截屏2021-03-10 17.14.25" style="zoom:80%;" />

## Multiprotocol Label Switching (MPLS)

### General Aspects

**MPLS**

- Based on label switching
- Originally: data plane optimization
- Standardized within the IETF
- Increasingly applied in larger autonomous systems
- Main Features
  - Fast forwarding (due to reduced amount of packet processing)
  - QoS support
    - Guarantees on latency and capacity, e.g., for voice traffic
  - Traffic engineering
    - Supports load balancing in order to optimize network utilization ...
  - Virtual private networks
    - Isolate traffic from other packets on the Internet
  - Multiple networks support
    - Usable on different network technologies, e.g., IP, ATM ...

👍 Advantages

- Clear separation of forwarding (label switching) and control (manipulation of label binding)
- Not limited to IP
- Support of metrics
- Versatile concept
- Scales

### Architecture, Components and Basic Operation

#### Architecture

![截屏2021-03-10 17.20.17](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.20.17.png)



#### Components

- **Label-switching router (LSR)**

  - MPLS-capable IP router

    - Can forward packets based on both, IP prefixes and MPLS labels 
    - Typically: IP for control plane and MPLS for data plane

  - Architecture:

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.35.34.png" alt="截屏2021-03-10 17.35.34" style="zoom:80%;" />

- **Label edge router (LER)**

  - Router at the edge of an MPLS domain
    - Each LSR with a non-MPLS capable neighbor is an LER 
    - Also called: label ingress router resp. label egress router
  - Classifies packets that enter the MPLS domain
    - [Forwarding equivalency class (FEC)](#forwarding-equivalence-class)

- **MPLS-Node**: General term for MPLS-capable intermediate systems, like LSRs

#### Forwarding Equivalence Classs

- Class of packets that should be treated **equally**
  - Same path through the network

  - Same QoS properties
- Basis for label assignment

- MPLS-specific term, roughly comparable to „flow“
- Example
  - Same address prefix and same type-of-service field 
  - Same IP addresses and same port numbers

  - VoIP traffic with destination address in subnet X
- Granularity
  - **Coarse-grained**: Important for quick forwarding and scalability 
  - **Fine-grained**: Important for differentiated treatment of packets or flows

**Example 1: Very fine granular FEC (“micro flow”)**

- A single TCP connection, identified by 5-tuple

  ![截屏2021-03-10 17.41.47](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.41.47.png)

**Example 2: data streams differentiation**

![截屏2021-03-10 17.42.35](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.42.35.png)

- Traffic engineering

  - Usage of different paths

  - Goals

    - Load balancing

    - Utilization of all available resources 
    - Prioritization of individual data streams

    (realized through separate virtual connections)

-  Support of quality of service

  - Different quality of service for different data streams

#### Label Switched Path

*Virtual* connection: Sequence of labels on a path through MPLS domain.

Example:

![截屏2021-03-10 17.45.30](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.45.30.png)



#### MPLS-Label

Encapsulation: **Between headers of layer 2 (Data Link layer) and layer 3 (Network layer)**

![截屏2021-03-10 17.47.06](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.47.06.png)

- Label: the label itself
- Exp: Bits for experimental usage
- S: Stack-bit
- TTL: Time-to-live

### Label Distribution

- **Label Binding**
  - Associate specific label to FEC
  - Stored in **label forwarding information base** 
    - Used as *incoming* label

- **Label distribution**
  - Label binding is distributed to neighboring routers
  - Stored in **label forwarding information base** 
    - Used as *outgoing* label

#### Types of Label Distribution

- “Roles” of a label-switching router

  ![截屏2021-03-10 17.56.41](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-10%2017.56.41.png)

  - **Downstream LSR**: *In* direction of data flow
  - **Upstream LSR**: *Against* direction of data flow

- **Unsolicited downstream**
  - Router generates label bindings as soon as it is ready to forward MPLS packets of the respective FEC
    - Upstream neighbors (according to IP routing): update forwarding tables
      - Label used as outgoing label
    - Non-upstream neighbors can store label for later use 
      - Quicker reactions on route changes
- **Downstream on demand**
  - Downstream router generates label binding on demand 
  - Upstream router has to request label binding for FEC

#### Label Distribution Protocol

##### **RSVP (Resource ReserVation Protocol)**

- 🎯 Goal: bandwidth reservation for end-to-end data streams 

- Soft state principle

  - Establish a session and periodically signal that session is still alive 
  - In case of failure state is automatically removed after some time

- Signaling

  ![截屏2021-03-11 13.07.14](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2013.07.14.png)

  - **Path message**
    - From sender to receiver

    - Find path to receiver

    - Each hop is recorded in the message
  - **Resv message**
    - From receiver to sender

    - Bandwidth reservation on return path

##### **RSVP-TE (Traffic Engineering)**

- Extension to RSVP to support label distribution

  - Many additional fields and functionality, e.g., fast reroute

- Signaling

  ![截屏2021-03-11 13.08.51](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2013.08.51.png)

  - **Path message**
    - From upstream LER to downstream LER 
    - Label request
    - Source route (“explicit route”) [optional]
  - **Resv message**
    - In response to path message

    - From downstream LER to upstream LER 
    - Label binding (hop-per-hop)

### Virtual Private Networks

- MPLS is useful for virtual private networks (VPNs)
- Use case: VPN traffic engineering
  - Customer with sites at different locations (e.g., different cities) wants to lease seamless “network” service
  - Requirements
    - Connect physically remote locations

    - Carry IP-based intranet traffic

    - Each customer has obtained an IP address block 
    - Guaranteed bandwidth / SLAs
  - Options
    - “Dark fibre” provider 
    - VPN backbone provider

#### Example: Private Networks over “Dark Fibre”

Suppose that three companies have sites at remote locations 

- Company A: Karlsruhe, Paris, Zürich

- Company B: Karlsruhe, Paris

- Company C: Karlsruhe, Paris

Each company runs a private network

- Different subnet for each site from customers IP address space
- Router connects site to other site(s)
- Data is transported over leased fiber optic cables (“dark fibre”) 
  - Capacity 155 Mbit/s, utilization marked in graph

![截屏2021-03-11 15.25.04](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.25.04-20210311155949753.png)

A provider uses MPLS to offer virtual private networks

- Has „points of presence (PoP)“ in all three cities

- Offers bandwidth at arbitrary rates

- Is cheaper than leasing fiber optic cables

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.39.48.png" alt="截屏2021-03-11 15.39.48" style="zoom:67%;" />

Question: Can the provider serve the need of all three companies?

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.40.12.png" alt="截屏2021-03-11 15.40.12" style="zoom:67%;" />

The answer is: YES! By utilizing **non-shortest paths**!

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.40.49.png" alt="截屏2021-03-11 15.40.49" style="zoom:67%;" />

We can achieve that using **VPNs implemented by Label Switching**

- Outer label: identifies path to LER

- Inner label: identifies VPN instance / customer

For company A:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.42.49.png" alt="截屏2021-03-11 15.42.49" style="zoom:67%;" />

- Inner label $5$: Indicates that this packet belongs to company A\
- Outer labels $2, 7, 1$: Label switching/Swapping

For company B:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.45.11.png" alt="截屏2021-03-11 15.45.11" style="zoom:67%;" />

For company C:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.45.33.png" alt="截屏2021-03-11 15.45.33" style="zoom:67%;" />

#### Label Distribution

Recall VPN example from above

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.40.49-20210311160013646.png" alt="截屏2021-03-11 15.40.49" style="zoom:67%;" />

- LSP for customer B (Karlsruhe $\rightarrow$ Paris) should take a “detour” over Zürich) to match bandwidth requirements

- Setup of LSPs over explicitly given route with [RSVP-TE](#rsvp-te-traffic-engineering)
  - Example: LSP “Karlsruhe to Paris over Zürich”
    - RSVP-TE signaling initiated at upstream LER (LER-KA) 
    - Note: LSPs are unidirectional!

How are the labels distributed?

- LER-KA1 (upstream) sends Path Message to LER-P (downstream).

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.58.04.png" alt="截屏2021-03-11 15.58.04" style="zoom:67%;" />

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.58.16.png" alt="截屏2021-03-11 15.58.16" style="zoom:67%;" />

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.58.28.png" alt="截屏2021-03-11 15.58.28" style="zoom:67%;" />

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2015.58.48.png" alt="截屏2021-03-11 15.58.48" style="zoom:67%;" />

- LER-P receives the Path Message and send Resv Message back.

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2016.06.48.png" alt="截屏2021-03-11 16.06.48" style="zoom:67%;" />
  
  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2016.07.23.png" alt="截屏2021-03-11 16.07.23" style="zoom:67%;" />
  
  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2016.07.34.png" alt="截屏2021-03-11 16.07.34" style="zoom:67%;" />
  
  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-11%2016.07.58.png" alt="截屏2021-03-11 16.07.58" style="zoom:67%;" />

{{% alert note %}} 

Notice that we have label $2$ in the 5th step, and also in the 8th step. This is valid because labels are **locally** distributed.

{{% /alert %}}



























## Resource

- MPLS - Multiprotocol Label Switching (2.5 layer protocol)

  {{< youtube BuIWNecUAE8>}}

