---
# Basic info
title: "Allgemeine Grundlagen"
linktitle: "Grundlage"
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
        parent: multiprozessoren
        weight: 1
---



## Parallele Architekturmodelle

### Multiprozessor mit gemeinsamem Speicher

- **UMA**: **U**niform **M**emory **A**ccess

- Bsp: **symmetrischer Multiprozessor (SMP), Multicore-Prozessor**

  - **Gleichberechtigter** Zugriff der Prozessoren auf die Betriebsmittel

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-10%2022.10.42.png" alt="截屏2020-07-10 22.10.42" style="zoom: 67%;" />

> **Uniform memory access** (**UMA**) is a [shared memory](https://en.wikipedia.org/wiki/Shared_memory_architecture) architecture used in [parallel computers](https://en.wikipedia.org/wiki/Parallel_computer). All the processors in the UMA model share the physical memory uniformly. In an UMA architecture, access time to a memory location is independent of which processor makes the request or which memory chip contains the transferred data.
>
> More see: [Uniform memory access](https://en.wikipedia.org/wiki/Uniform_memory_access)

### Multiprozessor mit verteiltem Speicher

- **NORMA**: **No** **R**emote **M**emory **A**ccess
- Bsp: Cluster

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-10%2022.11.55.png" alt="截屏2020-07-10 22.11.55" style="zoom: 67%;" />

### Multiprozessor mit verteiltem gemeinsamen Speicher

- **NUMA**: **N**on-**U**niform **M**emory **A**ccess
- CC-NUMA: **C**ache-**C**oherent **N**on-**U**niform **M**emory **A**ccess
  - Globaler Adressraum: Zugriff auf *entfernten* Speicher über load / store Operationen

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-10%2022.13.31.png" alt="截屏2020-07-10 22.13.31" style="zoom: 67%;" />



