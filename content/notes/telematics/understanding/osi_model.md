---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 201

# Basic metadata
title: "OSI Model"
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
        weight: 1

---

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-14%2015.16.23.png" alt="截屏2021-03-14 15.16.23" style="zoom:80%;" />

| Layer Nr | Layer Name   |
| -------- | ------------ |
| 7        | Application  |
| 6        | Presentation |
| 5        | Session      |
| 4        | Transport    |
| 3        | Network      |
| 2        | Data Link    |
| 1        | Physical     |



## Mnemonic

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-14%2015.18.03.png" alt="截屏2021-03-14 15.18.03" style="zoom:67%;" />

## How does Data Flows the OSI Model Layers?

Client makes request to server

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/OSI_client2server.gif" alt="OSI_client2server" style="zoom:67%;" />

Server response

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/OSI_server2client.gif" alt="OSI_server2client" style="zoom:67%;" />

The round trip of how data flows through all these seven layers on both sides is a **physical path**, on which data actually and physically flows. 

The OSI model also addresses another aspect how data flows on a **logical path**, layer to layer commnunication.

![截屏2021-03-14 15.39.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-14%2015.39.23.png)

| Layers       | Sender                      | Receiver                    |
| ------------ | --------------------------- | --------------------------- |
| Application  | generate data               | read data                   |
| Presentation | encrypt and compress data   | decrypt and decompress data |
| Session      |                             |                             |
| Transport    | choke up data into segments | put segments together       |
| Network      | make packets                | open packets                |
| Data Link    | make frames                 | open frames                 |
| Physical     |                             |                             |



## OSI Model Layer by Layer  

### Application Layer

- Non-technical: **user's application** (E.g. Chrome, Firefox )
- Technical: refers to application protocols
  - E.g. HTTP, SMTP, POP3, IMAP4, ...
  - Facilitate communications between application and operation system

- Application data is generated here

### Presentation Layer

- Provides a variety of coding and conversion functions on application data

  - Ensure that information sent from the application layer of the client could be understood by the application layer of the server

    $\rightarrow$ Try to translate application data into a certain format that every different system could understand 

- Main functions
  - Data conversion
  - Data encryption
  - Data compression
- Protocols
  - Images: JPEG, GIF, TIF, PNG, ...
  - Videos: MP4, AVI, ...

### Session Layer

- Establish, manage, and terminate connections between the sender and the receiver

- An intuitive example

  {{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/OSI_Session_Example.gif" caption="Telephone call is a good example to explain session layer: First establish the connection and start the conversation. Then terminate the session" numbered="true" >}}

### Transport Layer

- Accept data from Session layer
- Choke up data into segments
- Add header information 
  - E.g. destination port number, source port number, sequence number, ...

- Protocols: **TCP and UDP**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/OSI_transport.gif" alt="OSI_transport" style="zoom:67%;" />

### Network Layer

- Protocol: **Internet Protocol (IP)**
- Take segment from Transport layer and add extra header information
  - E.g. sender's and receiver's IP address
- Create packet

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/OSI_Network.gif" alt="OSI_Network" style="zoom:67%;" />

### Data Link Layer

- When IP packet arrives at this layer, more header information will be added to the packet
  - E.g. source and destination MAC address, FCS trailer

- Ethernet frames are created

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/OSI_DataLink.gif" alt="OSI_DataLink" style="zoom:67%;" />

MAC address is physical address for your Network Interface Card (NIC). At this layer, NIC has crucial job of creating frames on the sender side, and reading or destroying frames on the receiver side.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/MAC_address.gif" alt="MAC_address" style="zoom:67%;" />

### Physical Layer

- Accept frames from Data Linker layer and generate bits

- These bits are made of electrical impulses or lights

- Through the network media, the data travels to the receiver

  $\rightarrow$ It completes the whole journey of seven layers on the sender side

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/OSI_physical.gif" alt="OSI_physical" style="zoom:67%;" />



## Reference

- OSI Model 👍

  {{< youtube nFnLPGk8WjA>}}