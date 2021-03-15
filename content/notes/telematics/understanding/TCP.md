---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 205

# Basic metadata
title: "TCP"
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
        weight: 5

---

**TCP** = **T**ransmisson **C**ontrol **P**rotocol

## How TCP starts and closes session?

Three stages of TCP

![截屏2021-03-15 11.43.05](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2011.43.05.png)

1. [Session starting](#session-starting)
2. [Data transmission](#data-transmission)
3. [Session ending](#session-ending)

This three stages make TCP a connect-oriented and reliable protocol. 👏

Suppose a client wants to get web pages from a server. Three stages below will be gone through.

### Session Starting

![截屏2021-03-15 11.31.50](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2011.31.50.png)



**Three-way handshake** to start a session

1.  Client sends a single `SYN` packet to the server, asking for a session

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2011.13.10.png" alt="截屏2021-03-15 11.13.10" style="zoom:67%;" />

   > Client: Hi, server, do you want to talk?

2. Server replies with a `SYNACK` packet (The server acknowledges the client's request, and ask client for a  talk)

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2011.27.59.png" alt="截屏2021-03-15 11.27.59" style="zoom:67%;" />

   > Client: Hi, server, do you want to talk?
   >
   > Server: Yes, I want to talk. Do you want to talk?

3. The client replies with `ACK` packet.

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2011.30.19.png" alt="截屏2021-03-15 11.30.19" style="zoom:67%;" />

   > Client: Hi, server, do you want to talk?
   >
   > Server: Yes, I want to talk. Do you want to talk?
   >
   > Client: Yes, I want to talk.

### Data Transmission

After three-way handshake, connection is established. And data packets are going to be transferred.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2011.35.37.png" alt="截屏2021-03-15 11.35.37" style="zoom:67%;" />

During data transmission, TCP also guarantees that data is successfully received and resembled in a correct order.

### Session Ending

After server sends all packets to the client, a four-steps procedure is performed before the connection is closed

1. The Server sends `FINACK` packet to the client.

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2011.38.15.png" alt="截屏2021-03-15 11.38.15" style="zoom:67%;" />

   > Server: I am done. Can you hear me?

2. The client responsed with `ACK` package.

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2011.39.06.png" alt="截屏2021-03-15 11.39.06" style="zoom:67%;" />

   > Server: I am done. Can you hear me?
   >
   > Client: Yes, I got your message, I can hear you.

3. When the client completes download in the webpage, it sends `FINACK` to the server

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2012.11.00.png" alt="截屏2021-03-15 12.11.00" style="zoom:67%;" />

   > Server: I am done. Can you hear me?
   >
   > Client: Yes, I got your message, I can hear you.
   >
   > Client: I am done. Can you hear me?

4. The server responses with `ACK`

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2011.41.18.png" alt="截屏2021-03-15 11.41.18" style="zoom:67%;" />

After this, the session between them can be properly close, unless the client continues to ask for another webpage.

## TCP Three-way Handshake in Detail

Suppose the client wants to get web pages from the server. Before any web page transmission, TCP connection must be established through **three-way handshake**.

1. The client sends `SYN` segment to the server, asking for synchronization (synchronization means connection)

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2012.42.14.png" alt="截屏2021-03-15 12.42.14" style="zoom:67%;" />

2. The server replies with `SYN-ACK` (synchronization and acknowledgement)

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2012.43.43.png" alt="截屏2021-03-15 12.43.43" style="zoom:67%;" />

   - The server acknowledges the client's connection request
   - The server also asks the client to open a connection too.

3.  The client replies `ACK`, which is like "Yes". Then the two-way connection is established between them.

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/3_time_handshake.gif" alt="3_time_handshake" style="zoom:67%;" />

### More Technical View

1. The client sends a `SYN` segment with the initial sequence number `9001`

   - `ACK` is set to 0
   - `SYN` is set to 1

   ![截屏2021-03-15 12.50.33](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2012.50.33.png)

2. The server replies with `SYN-ACK`

   ![截屏2021-03-15 12.52.59](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2012.52.59.png)

   - The server's `SYN` is set to 1
   - `ACK` is set to 9002, which is the client's sequence number plus 1. 
     - By adding 1 to the client's sequence number, the server simply acknowledges the client connection request)
   - The server's segment has its own initial sequence number 5001

3. The client responses

   ![截屏2021-03-15 12.56.22](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2012.56.22.png)

   - `SYN` is set to 0 : There is NO more synchronization/connection request)
   - `ACK` is set to 5002: The client acknowledges the server connection request by increasing the server-side sequence number by 1 (5001 + 1 = 5002)
   - `Seq` is set to 9002: This is the second segment issued by the client at this point (9001 + 1 = 9002)

At this point, both client and server have agreed to open their connection to each other.

- Step 1 and 2: establish the connection from client to server
- Step 2 and 3: establish the connection from server to client 

$\rightarrow$ Two-way connection channel is established and they are ready to exchange their messages.



## TCP vs. UDP

**TCP** (**T**ransmission **C**ontrol **P**rotocol) and **UDP** (**U**ser **D**atagram **P**rotocol) are two protocols of the Transport layer (Layer 4) in the OSI model.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/TCP_UDP.gif" alt="TCP_UDP" style="zoom:67%;" />

|                           | TCP                         | UDP                                                          |
| ------------------------- | --------------------------- | ------------------------------------------------------------ |
| [Reliable](#reliablity)   | Yes                         | No                                                           |
| [Connection](#connection) | connection-oriented         | connectionless                                               |
| Character                 | reliable                    | faster, more efficient                                       |
| Usage                     | dominant transport protocol | <li> live streaming audio/video <li>DNS queries, DHCP or VoIP <li> Only a few applications use UDP |

### Reliablity

- When **TCP** delivers data segments to destination, the protocol makes sure 

  - each segement is received,
  - no error,
  - and segments are put together in a correct order

  $\rightarrow$ TCP is reliable 👍

- When **UDP** delivers data segments to destination, it does NOT guarantee, or even NOT care if the data reach the destination. Once the data is sent off, UDP says "Goodbye, and good luck!"

  $\rightarrow$ UDP is NOT reliable 🤪

### Connection

TCP: connection-oriented

- TCP uses three-way handshake to make sure the connection is established before data transmission.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2012.02.03.png" alt="截屏2021-03-15 12.02.03" style="zoom:67%;" />

- After data is delivered, TCP will follow a 4-step procedure to make sure every bit of data is delivered and received before clossing the connection.

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2012.02.30.png" alt="截屏2021-03-15 12.02.30" style="zoom:67%;" />

UDP: connectionless

​	<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-15%2012.05.16.png" alt="截屏2021-03-15 12.05.16" style="zoom:67%;" />

- No handshake to establish the connection
- No procedure to close the connection

### Example for Understanding TCP and UDP

TCP walks to a bar 

> TCP: I want a beer.
>
> Bartender: You want a beer?
>
> TCP: Yes, I want a beer.



UDP walks to a bar

> UDP: I want a beer. (He does NOT care if the bartender hears him or not)

UDP might never get a beer...Well, he's UDP. He doesn't care.













## Reference

#### How TCP starts and close session?

{{< youtube zlIHLnOigmA>}}

#### Three-time handshake

{{< youtube xMtP5ZB3wSk>}}

#### TCP vs. UDP

{{< youtube SLY4Ud53UGs>}}



