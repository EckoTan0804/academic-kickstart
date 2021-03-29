---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 105

# Basic metadata
title: "Software Defined Networks (SDNs)"
date: 2021-03-11
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
        weight: 5

---

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/SDN-SDN_summary.png" caption="SDN summary" numbered="true" >}}

## Basics and Architecture

### High Level View on Traditional IP Networks

Abstract view on an IP router

![截屏2021-03-12 10.22.31](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2010.22.31.png)

- Control plane
  - Exchange of routing messages for calculation of routes ...
  - Additional tasks, such as load balancing, access control, ...
- Data plane: Forwarding of packets at layer 3

Every router has control and data plane functions

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2010.40.20.png" alt="截屏2021-03-12 10.40.20" style="zoom:80%;" />

- Control plane: software running on the router
- Data plane
  - usually application-specific integrated circuits 
  - Can also be realized in software (virtual switches)

Control is **decentralized**.

🔴 Limitations: Limited flexibility for network operators

- Manufacturer-specific management interfaces

- Difficult (and often impossible) to introduce new functions 
- Complex, highly qualified operators required

- Expensive (at least for core routers)

### Current Trend: **Software-Defined Networks (SDN)** 

👍 Advantages

- Increase flexibility

- Decrease dependencies on hardware and manufactures 
- Commercial off-the-shelf switches (cheaper)

#### Characteristics

- **Separation** of control plane and data plane

  ![截屏2021-03-12 10.50.06](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2010.50.06.png)

  - Control functionality resides on a **logically centralized SDN controller**
    - Controller is executed on commodity hardware $\rightarrow$ Reduces need for specialized routing hardware
  - Data plane consists of **simple packet processors (SDN switches)**

- Control plane has ***global*** network view 
  - Knows *all* switches and their configurations 
  - Knows network *topology*

- Network is **software-programmable**
  - Functionality is provided by network applications (network apps) 
  - Different apps can realize different functionality

  - SDN controller can execute multiple apps in parallel

- Processing is based on **flows**

### Basic Operation

![截屏2021-03-12 11.08.24](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.08.24.png)

- Control functionality is placed on the SDN controller 
  - E.g., routing including routing table
- Forwarding table is placed on SDN switch
  - Called **flow table** in the context of SDN

- SDN controller programs entries in flow table according to its control functionality
  - Requires a protocol between controller and switch

- For every incoming packet in the SDN switch
  - Suited entry in flow table needs to be determined

#### Flows and Flow Table

- **Flows**: sequence of packets traversing a network that *share* a set of header field values
  - Here: Identified through **match fields**, e.g., IP address, port number

- Flow table contains, among others, **match fields and actions**
  - Matches select appropriate flow table entry

  - Actions are applied to all packets that satisfy a match
  - E.g.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.13.29.png" alt="截屏2021-03-12 11.13.29" style="zoom:80%;" />

- Flow rule
  - Decision of controller

  - Described in form of match fields, actions, switches

#### Flow Rule and Flow Table Entries

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.15.02.png" alt="截屏2021-03-12 11.15.02" style="zoom:80%;" />

1. Controller (more precise: app executed by controller) makes a **high level decision**, for example

   a)  Traffic for destination X has to be dropped

   b)  Connection between end system A and B has to go through switch S4

   c) ...

2. High level decision is represented in a certain format, i.e., as a set of **flow rules** in the form of match fields, actions and switches
3. Flow rules are transmitted (“installed”) to switches with the help of a communication protocol. They are stored as **flow table entries** in flow tables

#### Flow Programming

SDN provides two different modes

- ***Proactive* flow programming**

  Flow rules are programmed ***before*** first packet of flow arrives

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.19.45.png" alt="截屏2021-03-12 11.19.45" style="zoom:80%;" />

- ***Reactive* flow programming**

  Flow rules are programmed ***in reaction to*** receipt of first packet of a flow

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.20.21.png" alt="截屏2021-03-12 11.20.21" style="zoom:80%;" />

##### Three Important Interactions

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.26.11.png" alt="截屏2021-03-12 11.26.11" style="zoom:80%;" />

##### Example: Proactive Flow Programming

Scenario

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.28.13.png" alt="截屏2021-03-12 11.28.13" style="zoom:80%;" />

![截屏2021-03-12 11.35.40](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.35.40.png)

<details>

<Summary>Details</Summary>

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.42.08.png" alt="截屏2021-03-12 11.42.08" style="zoom: 67%;" />

</details>

##### Example: Reactive Flow Programming

Same scenario as above

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.43.41.png" alt="截屏2021-03-12 11.43.41" style="zoom:80%;" />

<details>

<Summary>Details</Summary>

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.44.19.png" alt="截屏2021-03-12 11.44.19" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.44.31.png" alt="截屏2021-03-12 11.44.31" style="zoom:67%;" />

</details>

##### Proactive vs. Reactive Flow Programming

| Flow Programming | Characteristics             | Delay? | Loss of controller connectivity |
| ---------------- | --------------------------- | ------ | ------------------------------- |
| **Proactive**    | coarse grained, pre-defined | No     | Does not disrupt traffic        |
| **Reactive**     | fine grained, on demand     | Yes    | New flows cannot be installed   |

- Proactive

  - Flow table entries have to be programmed before actual traffic arrives  
    - Usually coarse grained **pre-defined** decisions

    - Not always applicable 🤪

  - No additional delays for new connections

  - Loss of controller connectivity does not disrupt traffic

- Reactive
  - Allows fine grained **on-demand** control
    - Increased visibility of flows that are active in the network
  - Setup time for each flow $\rightarrow$ High overhead for short lived flows 
  - New flows cannot be installed if controller connectivity is lost

### SDN Architecture

![截屏2021-03-12 11.51.50](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2011.51.50.png)

- **Application Plane**
  - Network apps perform network control and management tasks 
  - Interacts via **northbound API** with control plane

- **Control Plane**

  - Control tasks are „outsourced“ from data plane to logically centralized control plane
    - E.g., standard tasks such as topology detection, ARP ...
  - More complex tasks can be delegated to application plane
    - E.g., routing decisions, load balancing ...

- **Data Plane**

  - Responsible for packet forwarding / processing
  - SDN switches are relatively simple devices
    - Efficient implementations in hardware (ASIC) or in software (virtual switches) 
    - Supports basic operations such as match, forward, drop
  - Interacts via **southbound API** with control plane

- Interfaces

  - **Northbound API**: between controller and network apps
    - Exposes control plane functions to apps

    - Abstract from details, apps can operate on consistent network view

  - **Southbound API**: between controller and switches 
    - Exposes data plane functions to controller 
    - Abstracts from hardware details
  - **Westbound API**: between controllers
    - Synchronization of network state information
    - E.g., coordinated flow setup, exchange of reachability information
  - **Eastbound API**: interface to legacy infrastructures
    - Usually proprietary

## SDN Workflow in Practice

### Workflow and Primitives

High level view:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.05.50.png" alt="截屏2021-03-12 12.05.50" style="zoom:80%;" />



In practice:

![截屏2021-03-12 12.08.56](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.08.56.png)

1. We need a piece of software (app) that realizes the new behavior

   `control_my_network.java`

2. We need primitives to assist with creating the app

   `import OFMatch, OFAction, ...`

3. We need a runtime environment that can execute our app

   ```bash
   $ ./myController --runApp control_my_network.java
   ```

4. We need hardware support for SDN in the switches

   **Flow table(s)**

#### Primitives for SDN Programming

🎯 Goal: From intended behavior to lower level flow rules

$\rightarrow$ This requires **SDN programming primitives**

Three important areas to cover

##### (1) Create and install flow rules

Sufficient for proactive use cases.

Example: Traffic with IP destination address `1.2.3.4` has to be forwarded to network B by switch S1

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.25.11.png" alt="截屏2021-03-12 12.25.11" style="zoom:80%;" />

Needed: App that implements the corresponding logic 

- Represent the decision as flow rules

- Program appropriate flow table entries into the switch

Suppose that we have `static_forwarding.java`

- Creates a new flow rule

- Sends the flow rule to S1

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.34.13.png" alt="截屏2021-03-12 12.34.13" style="zoom:80%;" />

<details>
<summary>Details</summary>
<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.34.44.png" alt="截屏2021-03-12 12.34.44" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.35.36.png" alt="截屏2021-03-12 12.35.36" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.36.01.png" alt="截屏2021-03-12 12.36.01" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.36.17.png" alt="截屏2021-03-12 12.36.17" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.36.52.png" alt="截屏2021-03-12 12.36.52" style="zoom:67%;" />

</details>

<br>

{{% alert note %}} 

Here we use a simple **pseudo programming language**

- Language used in practice depends on controller
- Different controllers support different languages: Java, Python, C, C++, ...

{{% /alert %}}

**Overview: Matches**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.43.27.png" alt="截屏2021-03-12 12.43.27" style="zoom:67%;" />

**Overview: Actions**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.43.56.png" alt="截屏2021-03-12 12.43.56" style="zoom:67%;" />

**Priorities**

- Priorities come into play if there are **overlapping** flow rules
  - No overlap = all potential packets can only be matched by at most one rule 
  - Overlap = at least one packet could be matched by more than one rule

- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.46.39.png" alt="截屏2021-03-12 12.46.39" style="zoom:67%;" />

  Assume that all rules are created with same **default priority (=1)**

  If two rules can overlap, priority has to be changed explicitly

  - Higher values = higher priority

  ![截屏2021-03-12 12.50.47](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.50.47.png)

**Multiple Flow Tables**

- SDN switches can support more than one flow table

  ![截屏2021-03-12 12.51.59](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2012.51.59.png)

- Using multiple tables has several benefits

  - Can be used to isolate flow rules from different apps
  - Logical separation between different tasks (one table for monitoring, one table for security, ...)
  - In some situation: less overall flow table entries

- Similar to single table case

  - `r.TABLE(x)`: specify the table for this rule
  - `r.ACTION('GOTO', y)`: specify processing continues in another table

- Avoid cycles: Can NOT go to lower flow table number 

  - `GOTO` from table x to table y $\Rightarrow$ y > x 

- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2013.04.07.png" alt="截屏2021-03-12 13.04.07" style="zoom:67%;" />

##### (2) React to data plane events

- `onPacketIn(packet, switch, inport)`
  - Called if the controller receives a packet that was forwarded via `r.ACTION('CONTROLLER')`
  - Parameters
    - `packet`: contains packet that was forwarded and grants access to its header fields
      - `packet.IP_SRC `
      - `packet.IP_DST `
    - `packet.MAC_SRC` 
      - `packet.MAC_DST` 
      - `packet.TTL`
      - ...
    - `switch`: the switch the packet was received at (e.g., S1)
    - `inport`: the interface the packet was received at (e.g., port 1)
  
- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2015.36.45.png" alt="截屏2021-03-12 15.36.45" style="zoom:80%;" />

  - Sketch

    1. Create a low priority flow rule that sends „all unknown packets“ to the controller

       ```java
       r.MATCH('*') // match on everything 
       r.ACTION('CONTROLLER') // send packet to controller 
       r.PRIORITY(0) // use lowest priority for this flow rule
       ```

    2. Use `onPacketIn()` to create and install flow rules on demand

<details>
<summary>Details</summary>
<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2015.41.49.png" alt="截屏2021-03-12 15.41.49" style="zoom:80%;" />
  
  

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2015.42.12.png" alt="截屏2021-03-12 15.42.12" style="zoom:80%;" />
	
<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2015.42.27.png" alt="截屏2021-03-12 15.42.27" style="zoom:80%;" />
	
<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2015.42.34.png" alt="截屏2021-03-12 15.42.34" style="zoom:80%;" />
	
</details>

##### (3) Inject individual packets

Handle individual packets from within the app

- Forward a packet that was sent to the controller
- Perform topology detection
- Active monitoring („probe packets“)
- Answer ARP requests

`send_packet(packet, switch, rule)`

- Injects a single packet into a switch

- Parameters

  - `packet`: contains the packet that should be injected

  - `switch`: the switch where the packet is injected

  - `rule`: a flow rule that is applied to this packet instead of default flow table

    processing (optional)

    - Only `rule.ACTION()` is allowed here 
    - No matches, no priorities

- Different from installing flow rules
  - Used for a single packet only
  - The flow table is not changed
  - Even if the `rule `parameter is present, this does NOT create a new flow table entry

Inject and process injected packet with a custom rule

- Directly attaches the actions to the injected packet
- Rule is only used for a single packet
- Flow table remains unchanged
- Advantages
  - Efficient
  - Consistent

Example

```java
newPacket = createNewPacket()
customRule = Rule()
customRule.ACTION('OUTPUT', 1)
send_packet(newPacket, switch, customRule)
```

#### Summary on Primitives

- **Entry point primitves**: Callbacks to implement custom logic

  - `onConnect(switch)`

    Called if a new control connection to switch is established

  - `onPacketIn(packet, switch, port)`

    Called if a packet was forwarded to the controller

- **Flow rule creation primitives**: Used to define flow rules

  - `Rule.MATCH()`

    Select packets based on certain header fields

  - `Rule.ACTION()`

    Specify what happens to a packet in the switch

  - `Rule.PRIORITY()`

    Specify the priority of the created flow rule

  - `Rule.TABLE()`

    Specify the flow table the rule should be applied to

- **Switch interaction primitives**: Used to handle flow rule installation and packet injection
  - `send_rule(rule, switch)`
    Installs a flow rule and creates the associated flow table entry in the switch
  - `send_packet(packet, switch)`
    Injects a single packet into a switch, process with existing flow table entries
  - `send_packet(packet, switch, rule)`
    Injects a single packet into a switch, process with custom rule

### Learning Switch Example

Goal: learn port-address association of end systems

- Switch receives packet and does not know destination address

  - **Floods** packets on all active ports

  - **Learns** "location" of the end system with this destination

    address

    	- Remembers that end system is accessible via this port 
    	- Entry in table `<MAC address, port, lifetime>`

- Switch receives packet and knows destination address 
  
  - Forwards packet via corresponding port

We can do the same with SDN: Learning switch app

- Observe packets by controller
- Derive locations of end systems
- Program forwarding rules to allow connectivity between end systems based on MAC addresses and port numbers

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.00.57.png" alt="截屏2021-03-12 23.00.57" style="zoom:67%;" />

#### Naïve Approach

- Send all packets to controller

- Controller looks at `INPORT` and **source MAC address**

- Controller creates rules based on these two pieces of information 
- Packets with unknown destination addresses are flooded to all ports

<details>
<summary>Implementation</summary>

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.16.13.png" alt="截屏2021-03-12 23.16.13" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.17.25.png" alt="截屏2021-03-12 23.17.25" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.18.13.png" alt="截屏2021-03-12 23.18.13" style="zoom:80%;" />

</details>

🔴 Problem

![截屏2021-03-12 23.27.47](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.27.47.png)

#### Version 2

- Delay rule installation until the destination address was learned (not the source address)
- Avoids installing rules „too early“

<details>
<summary>Implementation</summary>

![截屏2021-03-12 23.38.41](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.38.41.png)

![截屏2021-03-12 23.38.58](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.38.58.png)

![截屏2021-03-12 23.39.09](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.39.09.png)

</details>

Consider the example above:

![截屏2021-03-12 23.41.11](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.41.11.png)

![截屏2021-03-12 23.41.16](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.41.16.png)

🔴 Problem

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.43.04.png" alt="截屏2021-03-12 23.43.04" style="zoom:80%;" />

#### Version 3

- Only matching on destination address is not specific enough 
- **Use more specific matches**
- Makes sure that all end systems can be learned by controller

Implementation

![截屏2021-03-12 23.50.10](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.50.10.png)

Consider the example in Version 2:

![截屏2021-03-12 23.56.11](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.56.11.png)

🔴 Problem: flow table resources

- Needs N*N flow entries for N end systems 
- May exceed table capacity! 🤪

{{% alert note %}} 

The amount of flow table entries required is an important factor for usability and scalability.

{{% /alert %}}

#### Version 4

- Separate flow tables for learning and forwarding

  ![截屏2021-03-12 23.59.02](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-12%2023.59.02.png)

  - **Flow table FT1** matches on source address and forwards to controller, if address was not yet learned
  - **Flow table FT2** matches on destination address and forwards packet to destination (if learned) or floods packet (if not learned)

- Only 2*N rules for N end systems
- 🔴 Problem: Hardware often does not support multiple flow tables due to cost, energy or space constraints



## OpenFlow

### Rough Overview

- A standard for an SDN **southbound** interface
  - Defines the **interaction** between controller and switches 
  - Defines a **logical architecture** for SDN switches (flow table, ...) 
  - Defined by the Open Networking Foundation (ONF)

- Supports 
  - All basic structures and primitives discussed in previous section
    - Matches
    - Actions

    - Priorities

    - Multiple flow tables
    - Protocol mechanisms for 
      - Creating flow rules

      - Reacting to data plane events 
      - Injecting individual packets
  - More sophisticated features 
    - Group table
    - Rate limiting

### Structure

Provides a uniform view on SDN-capable switches

![截屏2021-03-13 00.07.45](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2000.07.45.png)

#### **Ports**

- Represent logical forwarding targets

- Can be selected by the output action

- Physical ports = hardware interfaces

- Reserved ports (special meaning)

  - `ALL`

    - Represents all ports eligible to forward a specific packet (= flooding); 
    - Ingress port is automatically excluded from forwarding

  - `IN_PORT`

    Always references ingress port of a packet (= send packet back the way it came)

  - `CONTROLLER`

    Forwarding a packet on this port sends it to the controller 

  - `NORMAL`

    Yields control of the forwarding process to the vendor-specific switch implementation

- Logical ports

  - Provide abstract forwarding targets (vendor-specific)

  - **Link aggregation**: Multiple interfaces are combined to a single logical port 
  - **Transparent tunneling**: Traffic is forwarded via intermediate switches

#### **Flow table**

![截屏2021-03-13 10.12.09](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.12.09.png)

- **Counters**

  The number of processed packets (counter)

- **Timeouts**
  - Maximum lifetime of a flow

  - Enables automatic removal of flows
- **Cookie**
  - Marker value set by an SDN controller 
  - Not used during packet processing 
  - Simplifies flow management
- **Flags**
  - Indicate how a flow is managed

  - E.g., notify controller when a flow is automatically removed

- **Pipeline Processing**

  - Multiple flow tables can be chained in a flow table **pipeline**
    - Flow tables are numbered in the order they can be traversed by packets 
    - Processing starts at flow table 0

    - Only “forward” traversal is possible $\rightarrow$ no recursion

    - Actions are accumulated in an action set during pipeline processing

  - Divided into **ingress** and **egress processing**

    ![截屏2021-03-13 10.15.12](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.15.12.png)

  - Example

    <details>
    <summary>Building an action set</summary>
    
    ![截屏2021-03-13 10.26.56](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.26.56.png)

    ![截屏2021-03-13 10.27.15](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.27.15.png)

    ![截屏2021-03-13 10.28.06](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.28.06.png)

    ![截屏2021-03-13 10.28.52](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.28.52.png)

    ![截屏2021-03-13 10.28.29](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.28.29.png)

  	![截屏2021-03-13 10.29.47](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.29.47.png)

  	![截屏2021-03-13 10.30.06](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.30.06.png)
    </details>

    

    



  

##### Ingress Processing

- Starts at flow table 0
- Initial action set is empty

![截屏2021-03-13 10.16.46](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.16.46.png)

##### Egress Processing

Optionally follows ingress or group table processing

- Egress flow tables must have higher table numbers than ingress tables $\rightarrow$ No return to ingress processing

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2011.54.11.png" alt="截屏2021-03-13 11.54.11" style="zoom:67%;" />

##### Group Tables

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.37.07.png" alt="截屏2021-03-13 10.37.07" style="zoom:80%;" />

Grout entry:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.38.12.png" alt="截屏2021-03-13 10.38.12" style="zoom:80%;" />

- Group tables represent additional forwarding methods (E.g., link selection, fast failover, ...)
- Group entries can be invoked from other tables via group actions
  - They are referenced by their unique group identifier

  - Flow table entries can perform group actions during ingress processing

- Effect of group processing depends on the **group type** and its **action buckets**

  - Action buckets

    - Each group references zero or more action buckets
      - Not every action bucket of a group has to be executed 
      - A group with no action buckets drops a packet
    - An action bucket contains a set of actions to execute (just like an action set)

  - Group types

    ![截屏2021-03-13 10.45.46](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2010.45.46.png)

    - **All**: executes all buckets in a group (E.g., for broadcast)
    - **Indirect**: executes the single bucket in a group
      - Indirect groups must reference exactly one action bucket

      - Useful to avoid changing multiple flow table entries with common actions
    - **Select**: selects one of many buckets of a group (E.g., select by round-robin or hashing of packet data)
    - **Fast failover**: executes **first live bucket** in a group
      - Each bucket is associated with a port that determines its liveliness

  - Example

    <details>
    <summary>Indirect Group Tables</summary>
    
    ![截屏2021-03-13 11.13.01](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2011.13.01-20210313111743768.png)
    
    🎯 Goal: Reroute flows to avoid forwarding via switch S2

    - Output ports specified in flow tables are subject to change

    - SDN controller must send multiple modify-state messages to SDN switches 
    
	- One message for each flow that needs to be updated 🤪
    
    ![截屏2021-03-13 11.14.33](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2011.14.33.png)

    Optimization

    - Use an **indirect group** to avoid sending multiple modify-state messages 
	- Redirect flows with identical forwarding behavior to that group
    - Modify the groups actions when forwarding behavior changes
    
    ![截屏2021-03-13 11.15.40](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2011.15.40.png)

    Advantage: Instead of modifying a great number of entries in flow table, we just need to modify one entry in group table!

    </details>
    
    


### Additional material on OpenFlow

#### Flow Table in OpenFlow

- Flow tables contain match/action-associations
  - Matches select the appropriate flow table entries
  - Actions are applied to all packets that satisfy a match
- **Table-miss flows** capture all unmatched packets
  - Enables reactive flow programming 
  - Corresponding flow table entry has *lowest* priority 
  - Synonym: **default flow**

- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2011.30.00.png" alt="截屏2021-03-13 11.30.00" style="zoom:80%;" />

#### Matches in OpenFlow

- Matches have **priorities**

  - Only the entry with the **highest priority** is selected 
  - Disambiguation of similar match fields

  ![截屏2021-03-13 11.32.10](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2011.32.10.png)

- Wildcard matching can be performed using **bitmasks** 
- Empty match fields match all flows

#### Actions in OpenFlow

- Basic functionality is simple: „determine what happens to a packet“

- In reality, OpenFlow makes a distinction between actions, action sets and more general instructions (linked to how the OpenFlow pipeline works)

  - **Action**
    - A concrete command to manipulate packets like „output on port“ or „push MPLS“
    - OpenFlow supports
      - **Output**: forwards a packet
      - **Set-field**: modifies a header field of a packet 
      - **Push-tag**: pushes a new tag onto a packet
      - **Pop-tag**: removes a tag from a packet
      - Drop a packet: Implicitly defined when no output action is specified

  - **Action set**

    - Every packet has its own ActionSet while processed

    - Changes to the packet can be stored in the set / deleted from the set 
    - Actual changes are applied when processing ends

    - Set is carried between flow tables (in one switch)
    - An action set contains **at most one action of a specific type**
      - Previous instances are overwritten
      - An action set may contain multiple set-field actions
    - Execution proceeds in a well-defined order
    - Modifications to the action set
      - **write-actions**: writing new actions to a set
      - **clear-actions**: Removing all actions from the set

    (Check out the example in [Flow Table](#flow-table))

  - **Instructions**
    - Control how packets are processed in the switch
    - Each flow table entry is associated with a set of instructions 
      - Change the packet immediately (`apply`-action)

      - Change the action set

      - Continue processing in another table (`goto`-table command)

#### OpenFlow Channel

- Connects each switch to a controller

- Provides the southbound API functionality of an OpenFlow switch 

  - **Management** and **configuration** of switches by controllers
  - **Signaling of events** from switches to controllers
  - **Monitoring** of liveliness, error states, statistics, ...

  - **Experimentation**

- Multiple channels to different controllers can be established

- Three message types 

  - **Controller-to-Switch messages**

    - Inject controller-generated packets (**packet-out** message)

    - Modify port properties or switch table entries (**modify-state** message) 
    - Collect runtime information (**read-state** message)

  - **Asynchronous messages**

    - **Packet-in** message transfers control of packet to the controller
    - State changes signaled by switches

  - **Symmetric messages**

    - Handle connection setup and ensure correct operation
      - **Hello**: exchanged on connection startup (e.g., indicate supported versions) 
      - **Echo**: verify lifelines of controller-switch connections
      - **Error**: indicate error states of the controller or switch

    - **Experimenter** messages can offer additional functionality

#### Meter Tables

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.00.07.png" alt="截屏2021-03-13 12.00.07" style="zoom:80%;" />

- Meter table entry

![截屏2021-03-13 12.01.10](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.01.10.png)

- Meters measure and control the **rate** of packets and bytes 
  - They are managed in the meter table

  - Each meter has a unique meter identifier

- Meters are invoked from flow table entries through the meter action 
  
- When invoked, each meter keeps track of the measured rate of packets
  
- One of several **meter bands** is triggered when the measured rate exceeds that bands target rate

- Meter bands

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.02.29.png" alt="截屏2021-03-13 12.02.29" style="zoom:80%;" />

  - Packet processing by a meter band depends on its **band type** 
    - **DSCP remark**: implements differentiated services
    - **Drop**: implements simple rate-limiting
  - **Rate** and **burst** determine when a band is executed
  - Band types may have additional type-specific **arguments**



## The Power of Abstraction

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.08.26.png" alt="截屏2021-03-13 12.08.26" style="zoom:80%;" />

### Different Abstractions for Different Apps

Controller can provide different abstractions to network apps

- Apps should not deal with low level / unnecessary details

- Apps only have an abstract view of the network

- Global view of controller can be different from abstract view of an app

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.10.01.png" alt="截屏2021-03-13 12.10.01" style="zoom:80%;" />

### Examples

#### "Big Switch Abstraction"

Consider a security application that manages access control lists 

- Controls the access of end systems E1, ... En to services S1, ... Sm

- Details such as the exact position of an end system / service are not required for the application $\rightarrow$ Can be hidden in the abstraction

![截屏2021-03-13 12.11.22](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.11.22.png)

#### Network Slicing

Consider a network that has to be **virtualized between multiple customers**, e.g., Alice and Bob

- Alice is only allowed to utilize S1, S2, and S3

- Bob is only allowed to utilize S2, S3, S5, and S6

- Both customers get an individual (full-meshed) view of the network
  - This is often called a network **slice**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.12.34.png" alt="截屏2021-03-13 12.12.34" style="zoom:80%;" />



## 🔴 SDN Challenges

### Controller connectivity

- SDN requires connectivity between controller and switches

- Two different connectivity modes  

  - **Out-of-band**

    - **Dedicated** (physical) control channel for messages between controller and switch

      ![截屏2021-03-13 12.41.52](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.41.52.png)

    - Cost intensive

  - **In-band**

    - Control messages use **same** channel as “normal” traffic (data)

      ![截屏2021-03-13 12.51.35](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.51.35.png)

    - Multiple applications can configure switch

### Scalability

- Logically centralized approach requires powerful controllers $\rightarrow$ Size / load of bigger networks can easily overload control plane 🤪

- Important parameters with scalability implications

  - Number of remotely controlled switches

  - Number of end systems / flows in the network

  - Number of messages processed by controller
  - Communication delay between switches and controller

- Possible solution: **Distributed controllers**

  ![截屏2021-03-13 12.53.35](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2012.53.35.png)

### Consistency

- Network view must remain **consistent** for applications

  - Synchronize network state information
  - Done via the **westbound** interface

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2016.37.25.png" alt="截屏2021-03-13 16.37.25" style="zoom:80%;" />

- Controller directly applies internal operations (inside partition) and notifies remote controllers of relevant changes of the network

  - E.g C1 applies internal operations in Partition 1 and then notifies C2 of the change.

- Apps can perform data plane operations on remote switches 

  - Apps operate on a consistent network view
  - Operations are delegated to responsible SDN controller

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2016.45.17.png" alt="截屏2021-03-13 16.45.17" style="zoom:80%;" />

- Note: Control plane with multiple controllers is a **distributed** system

  - Desirable properties

    - **Consistency**

      System responds identically to a request no matter which node receives the request (or does not respond at all)

    - **Availability**

      System always responds to a request (although response may not be consistent or correct)

    - **Partition tolerance**

      System continues to function even when specific messages are lost or parts of the network fail

  - **CAP theorem**
    - It is impossible to provide (atomic) consistency, availability and partition tolerance in a distributed system all at once

    - Only **two** of these can be satisfied at the same time

### Data plane limitations

- Flow Table Capacity
- Flow Setup Latency

## SDN Use Cases

- Google B4
- Defense4All
- VMWare NSX

## Tools

### Controller Platforms

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2016.51.31.png" alt="截屏2021-03-13 16.51.31" style="zoom:67%;" />

### Virtual Switches

- Core component in modern data centers
- Used as **“virtual” Top-of-Rack** switches



## Flow Programming Example 

This example is taken from HW09.

![截屏2021-03-13 22.25.12](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2022.25.12.png)

#### Describe the functionality that is implemented by `app_1.java`

The application has proactive and reactive parts

- Proactive: `onConnect()`
  - `r1`: Forward all packets whose IP destination address belongs to `28.0.0.0/8` to port 1 (i.e. network N1)
  - `r2`: Default rule, drops everything
  - `r3`: Send packets from N1 to controller

- Reactive `onPacketIn()`
  - If a packet is sent to controller by `r3`, check whether the MAC address is valid. If valid, then forward to port 4 (i.e, network N2). Otherwise drop.

#### What port is connected to the Internet in the given example?

A reasonable assumption here is that N1 is the internal network (because the application can check source validity with MAC addresses) and N2 is the Internet (i.e., the answer is port 4)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2022.32.30.png" alt="截屏2021-03-13 22.32.30" style="zoom:80%;" />

#### Why `r2.PRIORITY(0)` is required?

`r2.PRIORITY(0)` is required, because `r2` is the default rule in this case

- Default rules usually have `*` match 
- 0 is the lowest priority (lower than the default priority = 1)

#### why `r1.PRIORITY(2)` is required?

`r1.PRIORITY(2)` is required to enforce that the there are no rule overlaps

- With default priority on `r1` , `r1` and `r3` would overlap if a packet from N1 is sent with destination address in 28.0.0.0/8

#### Draw a sequence diagram illustrating the processing of the six consecutive packets P1 - P6 shown below. The diagram should contain the two networks (N1, N2), the switch (S) and the controller (C). Mark the arrows with send_rule, packet_in and send_packet

![截屏2021-03-13 22.37.00](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-13%2022.37.00.png)

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/SDN-HW09.png" caption="Solution" numbered="true" >}}

