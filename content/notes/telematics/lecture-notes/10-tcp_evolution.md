---

# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 110

# Basic metadata
title: "TCP Evolution"
date: 2021-03-19
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
        weight: 10

---

![截屏2021-03-21 17.21.19](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2017.21.19.png)

## TCP Extensions

### TCP Options: Basics

#### TCP Header

![截屏2021-03-21 17.25.54](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2017.25.54.png)

#### TCP Options

- 🎯 Goal: Flexibility for new developments
- TCP header field
  - Each option is coded in **TLV format (Type-Length-Value)**
  - Has variable but **limited** length
    - **number of options is limited (max. 40 bytes)**
    - TCP header length at most 60 bytes in total (incl. options)

- **TLV format**

  ![截屏2021-03-21 17.31.16](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2017.31.16.png)

  - Multiple of 32 bit words (If not padding is needed)

  - Type

    - [Selective acknowledgements](#option-selective-acknowledgements)

    - Time stamps

    - [Window scaling](#option-window-scaling)

    - Maximum segment size 
    - Multipath TCP

    - TCP fast open
    - ...

  - Length: Length of option
  - Value: Option data

### Option Selective Acknowledgements

- TCP uses **cumulative acknowledgements**

  - 👍 Pro: Very robust against loss of ACK segments

  - 👎 Cons: Inefficient loss recovery

    - Sender can only learn about a single lost segment per RTT

    - Consequently

      - Fast retransmit/fast recovery can only recover one lost segment

        per RTT

      - Multiple losses often lead to retransmission timeouts and head-of-line blocking

- Improvement: **selective acknowledgements** (SACK)

  - Also acknowledge “out-of-order” data
  - Implemented as TCP option

- 💡 Idea: **Separately acknowledge continuous blocks of out-of-order data**

- Usage of SACK option negotiated during connection establishment

  ![截屏2021-03-21 17.39.41](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2017.39.41.png)

- SACK option format

  ![截屏2021-03-21 17.40.45](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2017.40.45.png)

  - Typically, only 2-4 blocks can be “SACKed” in one segment

- Case

  ![截屏2021-03-21 18.16.04](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2018.16.04.png)

  Handling:

  - Use first entry of SACK option to report **new** information
  - Use subsequent entries of SACK option for redundancy Used for redundancy, 
    - if prior ACKs were lost

    - Should repeat most recently sent first blocks

- Different alternatives

  ![截屏2021-03-21 18.17.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2018.17.23.png)

- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2018.17.42.png" alt="截屏2021-03-21 18.17.42" style="zoom:67%;" />

### Option Window Scaling

- Header field receive window remains unchanged (16 bit)
- **Scaling factor can be changed**
  - E.g., measure window size in 32 bit words instead of bytes
- Option is negotiated during connection establishment 
  - Within SYN and SYN/ACK segments
- Scaling factor remains unchanged during lifetime of a TCP connection

### Extension SYN Cookies

## Multipath TCP (MPTCP)

- Motivation

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2021.37.30.png" alt="截屏2021-03-21 21.37.30" style="zoom:67%;" />

- 🎯 Goal: Extension of TCP for parallel usage of multiple paths **within a single TCP connection**
  - Improves reliability 
  - Increases performance
- Important requirements
  - **Application compatibility**
  - **Network compatibility**
- Challenges
  - **Middleboxes**

### Connection vs. Subflow

- **MPTCP connection**
  - Communication relation between sender and receiver 
  - Consists of one or multiple **MPTCP subflows**
- **MPTCP subflow**
  - Flow of TCP segments operating over an individual path
  - Started and terminated like a „regular“ TCP connection
    - Started with 3-way handshake

    - Closed with FIN or RST
  - Can be dynamically added and removed to/from an MPTCP connection

### Embedding into Protocol Stack

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2021.43.10.png" alt="截屏2021-03-21 21.43.10" style="zoom:67%;" />

### Connection Establishment

3-way handshake of TCP

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2021.44.49.png" alt="截屏2021-03-21 21.44.49" style="zoom: 67%;" />

TCP option `MP_CAPABLE`

- `X`, `Y`: token for client and server
  - Identification for subsequent addition/removal of subflows

### Adding a Subflow

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2021.47.59.png" alt="截屏2021-03-21 21.47.59" style="zoom:67%;" />

TCP option `MP_JOIN`

- 3-way handshake of TCP
- Use tokens exchanged during MPTCP connection establishment

### Sequence Numbers

Each MPTCP segment carries **two** sequence numbers

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2021.57.57.png" alt="截屏2021-03-21 21.57.57" style="zoom:67%;" />

- **Data sequence number** for overall MPTCP connection
- **Subflow sequence number** for individual flow
  - Each subflow has coherent sequence numbers without „holes“

### Congestion Control

- 🎯 Goals of MPTCP
  - **Improve throughput**

    Multipath flow should perform at least as well as a single path congestion control would on the best available path

  - **Do not harm**

    Multipath flow should not take up more capacity from any of the resources shared than if it were a single flow

  - **Balance congestion**

    A multipath flow should have as much traffic as possible off its most congested paths

- Congestion Control algorithm only applies to increase phase of congestion avoidance
  - Unchanged: slow start, fast retransmit, fast recovery and multiplicative decrease
- Different congestion windows
  -  $CWnd\_i$ per subflow $i$
  - $CWnd\_{total}$ per MPTCP connection (multipath flow)

- Assumption: Congestion window maintained in **bytes**
- Basic approach: **Couple** congestion control of different subflows

- **Linked increase** (congestion avoidance)

  For each ACK received on subflow $i$, increase $CWnd\_i$ by
  $$
  \min \left( \underbrace{\frac{\alpha * \text { bytes }\_{\text {acked }} * M S S\_{i}}{C W n d_{\text {total }}}}\_{\text{ Increase for multipath subflow }}, \underbrace{\frac{\text { bytes }\_{\text {acked }} * M S S\_{i}}{C W n d\_{i}}}\_{\text{ Increase „regular“ TCP would get in same scenario }}\right)
  $$
  (any multipath subflow cannot be more aggressive than a TCP flow in the same circumstances (do not harm))

  - $\alpha$: Describes **aggressiveness** of multipath flow
    $$
    \alpha=C W n d\_{\text {total }} \cdot \frac{\max \_{i}\left(\frac{C W n d\_{i}}{R T T\_{i}^{2}}\right)}{\left(\sum \frac{C W n d\_{i}}{R T T\_{i}}\right)^{2}}
    $$
    

## TCP in Networks with High BDP

### Scalability Issues

- It can take very long until the available data rate is fully utilized

- Cause

  - Very conservative behavior of congestion avoidance

    - Congestion window grows by one MSS per RTT
    - Slow window growth in congestion avoidance causes low average data rate

    ➡️  NOT efficient in networks with high bandwidth-delay products

- Require **faster increase** of the congestion window in congestion avoidance

### Faster Increase of Congestion Window

- 🎯 Goals
  - High resource utilization in networks with high bandwidth delay product 
  - Quick reactions to changes of the situation within the network

  - Fairness with respect to other TCP variants
- Different types of fairness
  - **intra protocol fairness**
    - All senders use **same** TCP variant
    - Goal: All flows should achieve **same** data rate
  - With new TCP variants: **inter protocol fairness**
  - Furthermore: **RTT fairness**
    - Fairness among TCP flows with different RTTs

#### CUBIC TCP 

- 🎯 Goals

  - Provide simple algorithm for networks with high bandwidth-delay product

  - **TCP-friendly**

    Behaves like standard TCP (i.e., TCP Reno) in networks with short RTTs and small bandwidth

  - **Congestion avoidance**

    Applies **cubic** function instead of linear window increase

  - Performance should not be worse than TCP Reno

- In comparison to TCP Reno

  - Better RTT fairness (Window growth independent of RTT)
  - Better scalability to high data rates

- Currently default congestion control in all major operating systems

**Congestion Window Increase**

- **Independent** from RTT

  - Use of actual time $t$ that has passed since last congestion incident. I.e. Window growth depends on time between consecutive congestion events

  - Apply **cubic** function
    $$
    W(t)=C(t-K)^{3}+W_{\max } \quad \text { with } \mathrm{K}=\sqrt[3]{\frac{W_{\max }(1-\beta)}{C}}
    $$

    - $C$: predefined constant that determines aggressiveness of increase
    - $W\_{max}$: congestion window size at latest congestion incident
    - $K$: time period that it takes to increase current window to $W\_{max}$ (in case of no further congestions)
    - $\beta$: multiplicative decrease of congestion window
      - $\beta = 0.5$ for TCP-Reno
      - $\beta = 0.7$ for CUBIC TCP

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2023.21.53.png" alt="截屏2021-03-21 23.21.53" style="zoom:67%;" />

**Congestion Window over Time**

Example

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2023.23.36.png" alt="截屏2021-03-21 23.23.36" style="zoom:80%;" />

**Three CUBIC Modes**

- **TCP-friendly region**

  - Ensures that CUBIC achieves at least same data rate as standard TCP in networks with small RTT

  - Observation: in networks with small RTTs, Cubic ́s congestion window grows *slower* than with TCP Reno

  - **Approach: “emulation” of TCP Reno (which uses AIMD)**

  - $AIMD(\alpha, \beta)$

    - $\alpha$: additive increase factor
      $$
      W = W + \alpha
      $$

    - $\beta$: multiplicative decrease factor
      $$
      W = \beta \cdot W
      $$

    TCP Reno uses $AIMD(1, \frac{1}{2})$

  - TCP-fair increment
    $$
    \alpha=3 \cdot \frac{1-\beta}{1+\beta}
    $$

    - Achieves same $W\_{avg}$ as $AIMD(1, \frac{1}{2})$

    - Average data rate of AIMD
      $$
      W\_{avg} = \frac{1}{R T T} \sqrt{\frac{\alpha \cdot(1+\beta)}{2 \cdot(1-\beta) \cdot p}}
      $$

      - $p$: loss rate

  - Window size of emulated TCP at time $t$
    $$
    W\_{T C P}=W\_{\max } \cdot \beta+\frac{3 \cdot(1-\beta)}{1+\beta} \cdot \frac{t}{R T T}
    $$

  - Recall window size of TCP cubic
    $$
    W(t)=C(t-K)^{3}+W_{\max }
    $$

  $\Rightarrow$ Rule

  - $W\_{Cubic} < W\_{TCP}$, then $CWnd$ is set to $W\_{TCP}$ each time an ACK is received
  - otherwise, $CWnd$ is set to $W\_{Cubic}$ each time an ACK is received

- **Concave region**: $CWnd < W\_{max}$ and not in TCP-friendly region

  - For each received ACK
    $$
    CWnd = CWnd+\frac{W\_{cubic}(t+R T T)-CWnd}{C W n d}
    $$

- **Convex region**: $CWnd > W\_{max}$ and not in TCP-friendly region
  - $CWnd$ is increased very carefully
  - searching for new 𝑊𝑚𝑎𝑥

## TCP and Response Time

### Basic Issue

- **Response time**

  - Time between initiation of a TCP connection and receipt of the requested data

  - Important components

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-22%2017.27.37.png" alt="截屏2021-03-22 17.27.37" style="zoom:67%;" />

    - Handshake of TCP connection establishment 
    - Slow start

    - Transmission of the object

  - Macroscopic Model

    - Response time without applying congestion control

      ![截屏2021-03-22 17.28.54](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-22%2017.28.54.png)
      - After 1st RTT: Client sends object request

      - After 2nd RTT

        - Client begins to receive object data

        - Receiver needs
          $$
          t = \frac{\text{object size } O}{\text{data rate } D}
          $$

      $\Rightarrow$ lower bound:
      $$
      \text{Response time} \geq 2 RTT + \frac{O}{D}
      $$
      ( With small objects, response time dominated by $RTT$s)

- Used Variables
  - $RTT$: round trip time [Seconds]
  - $MSS$: maximum segment size [bit]
  - $W$: Size of congestion window [MSS], given as multiples of MSS
  - $O$: Size of object that has to be transferred [bit]
  - $D$: Data rate [bit/s]

- Observation

  - $RTT$s have significant influence on response time
  - On connection establishment: 2 $RTT$𝑠 until reception of object begins
  - During object transmission
    - Small windows create pauses: waiting for ACKs

  - Majority of TCP connections in the Web has short lifetime

    $\rightarrow$ Slow start has significant impact on response time

- 🎯 Goals

  - Avoid „empty“ RTTs without data transport 
  - Reduce RTTs needed for slow start

### Bigger Initial Congestion Window

**💡 Idea: Increase initial congestion window (IW)**

- at least 10 segments, thus, about 15 Kbytes

### TCP Fast Open

- 🎯 Goal: Reduce delays that precede the transmission of an object

- **TCP Cookie**

  - Goal

    - Avoid DoS attacks

    - Disallow sending data within first SYN segment of first connection establishment to a server

    - Establish cookie for subsequent connections

  - Use cookie $\rightarrow$ avoid state keeping at server

  - Basic steps

    1. Client requests TFO cookie from server

       <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-22%2017.40.26.png" alt="截屏2021-03-22 17.40.26" style="zoom:67%;" />

    2. Client uses TFO cookies in subsequent TCP connections

       <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-22%2017.40.45.png" alt="截屏2021-03-22 17.40.45" style="zoom:67%;" />

### HTTP/2



### QUIC