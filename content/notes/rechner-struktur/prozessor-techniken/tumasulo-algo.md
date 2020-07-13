---
# Basic info
title: "Tumasulo Algorithm"
Linktitle: Tumasulo Algorithm
date: 2020-07-08
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Vorlesung", "Zusammenfassung", "Rechnerstruktur"]
categories: ["Computer Structure"]
toc: true # Show table of contents?

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: ""
share: false  # Show social sharing links?
featured: true
lastmod: true

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
    rechner-struktur:
        parent: prozessor-techniken
        weight: 6
---

## Tutorials

{{< youtube y-N0Dsc9LmU >}}



## Dynamic Scheduling

 💡 Idea: Get rid of stall cycles by allowing instructions to execute **out-of-order**

Road analogy:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-12%2021.12.50.png" alt="截屏2020-07-12 21.12.50" style="zoom:80%;" />

- An in-order-processer is like a road with one lane
  - If one car stops, all car behind it have to stop
- An out-of-order-processor is like a road with temporarily multiple lanes
  - If one car stops, the car behind it can take over



## Tumasulo's Algorithm

🎯 Goal: High Floating-Point (FP) performance without special compilers

### Key Structures

Consider following instructions:

 ```assembly
DIV.D F0, F2, F4
ADD.D F10, F0, F8
SUB.D F12, F8, F14
 ```

> `ADD.D` depends on `DIV.D` (true dependency on `F0`, Read-After-Write)
>
> - In-order execution: `DIV.D` --> `ADD.D` --> `SUB.D`. There'll be a lot of stall cycles due to data dependency
> - Out-of-order execution: We want `SUB.D` to proceed when `ADD.D` is waiting for `DIV.D`

- To allow `SUB.D` to proceed, we need to buffer `ADD.D` somewhere
  - In Tumasulo's Algorithm these buffers called **Reservation Stations (RSs)**
- To allow `ADD.D` to proceed when its operands become available, RSs must be informed when result available
  - In Tumasulo's Algorithm results are broadcasted to all RSs on **Common Data Bus (CDB)**

### Tumasulo Pipeline Phases

{{% alert note %}} 

- **Pipeline phase**: may takes **several** clock cycles
- **Pipeline stage**: always takes **SINGLE** clock cycles

{{% /alert %}}

1. **IF**: fetch next instruction into FIFO queue of pending instructions
2. **Issue**
   1. Get next instruction from head of instruction queue
   2. If there is a matching RS (i.e., no structural hazard), then issue instruction to RS
      - Write operand values if they are currently in registers
      - Otherwise, write identifiers of RSs that will produce operands
3. **Execute**
   - When all operands available (i.e., no RAW hazard) and Functional Unit (FU) are free, then execute
   - If not, monitor CDB for result to become available

4. **Write result**
   - Write Result on CDB to all awaiting RSs and register file
   - Mark RS free so that it can be used for another instructions

### Tumasulo-based FP Unit

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-12%2023.17.47.png" alt="截屏2020-07-12 23.17.47" style="zoom: 67%;" />

### Reservation Station (RS) Structure

Each RS has 7 fields:

- `op`: Operation to perform
- `RS1`: RS that will produce 1st operand 
  - `0`: indicates that operand is available
- `RS2`: RS that will produce 2nd operand 
- `val1`: Value of 1st operand
- `val2`: Value of 2nd operand
- `Imm/addr`: holds immediate or effective address
- `busy`: 
  - `1`: RS is occupied 
  - `0`: RS is free

#### Example

- Add
- 1st operand being produced by RS `Mul2`
- 2nd operand available in register with value 12.55

Then the content of RS is

| op   | RS1  | RS2  | Val1 | Val2  | Imm/addr | busy |
| ---- | ---- | ---- | ---- | ----- | -------- | ---- |
| add  | Mul2 | 0    | n/a  | 12.55 | n/a      | 1    |

### Register Structure

Each register also has field `RS`:

- RS ID that will produce this value
- Blank/0 if not applicable

#### Example

| Register | RS   | Value |
| -------- | ---- | ----- |
| F0       |      |       |
| F1       | Mul1 |       |
| F2       |      |       |
| F3       | Add2 |       |

- The `RS` field of F0 and F2 are blank, which means currently they contain valid value
- F1 is currently being produced by RS `Mul1`
- F3 is currently being produced by RS `Add2`



## Tomasulo's Pipeline Phase Details

### Issue 

- If a matching RS is available -> Issue FP instruction
- If source operands
  - currently available in the register -> It is issued together with the operand value
  - currently "in-flight" being produced -> The instruction is linked with the RS that will produce this operand value

#### Example

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-13%2011.44.54.png" alt="截屏2020-07-13 11.44.54" style="zoom:80%;" />

`MUL.D F0, F1, F2` is at the head of the Instruction Queue 

- Register `F2` is available and contains the value 0.2
- Register `F1` is being produced by RS `Add2`

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-13%2011.47.35.png" alt="截屏2020-07-13 11.47.35" style="zoom:80%;" />

The multiply instruction will issue to RS `Mul2` 

- The first operand will be produced by RS `Add2`
- The second operand is already available and equals 0.2

As the multiply writes register `F0`, the `RS` field of register `F0` has been set to `Mul2` since this is the RS that will produce it.

### Execute

- Execute when all operands are available (no RAW hazard)
- Several instructions may become ready at same time
  - For FP RS, order of execution is **arbitrary** (usually **FIFO**)
  - Load/Stores are executed **in-order**

### Write Result

When Functional Unit (FU) has produced the result , write it on CDB and from there to any RS and register waiting on it

#### Example

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-13%2011.57.13.png" alt="截屏2020-07-13 11.57.13" style="zoom:80%;" />

- The RS `Add2` is ready to execute 

- RS `Mul2` is waiting for this result.
- This result needs to be written to register `F1`

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-13%2012.01.07.png" alt="截屏2020-07-13 12.01.07" style="zoom:80%;" />

- The subtracted instruction in RS `Add2` is executed.
- It writes its result 2.0 onto the CDB together the identifier RS `Add2` that has produced it

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-13%2012.05.20.png" alt="截屏2020-07-13 12.05.20" style="zoom:80%;" />

- The result is broadcasted on CDB to all the RS and registers

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-13%2012.06.11.png" alt="截屏2020-07-13 12.06.11" style="zoom:80%;" />

- The first operand of RS `Mul2` is now available and equals 2.0
- Register `F1` has been set to 2.0 and its `RS` field has been cleared



## Features of Tomasulo

- **Register renaming** (via RSs)

  - Elimination of name dependencies, i.e., WAR & WAW

    (RS + Register File = *virtual* rgister set)

  - Distributed RS: allow operand forward to multiple RSs in 1 cycle

    - For centralized Register File: sequential access if only 1 port 🤪

- **Bypassing / forwarding**
  - A result is directly forwarded from execution unit to multiple RSs via **CDB**

- **In-order issue/dispatch**

- **Out-of-order execution**

- **Out-of-order instruction retiring/completion** (No precise exception)

## Tumasulo Example

{{< youtube YH2fFu-35L8 >}}