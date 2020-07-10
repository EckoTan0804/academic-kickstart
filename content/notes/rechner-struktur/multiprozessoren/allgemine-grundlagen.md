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



## Parallele Programmiermodelle

- **Abstraktion** einer parallelen Maschine, auf der der Anwender sein Programm formuliert

- Spezifiziert：
  - wie Teile des Programms parallel abgearbeitet werden, 
  - wie Informationen ausgetauscht werden
  - welche Synchronisationsoperationen verfügbar sind, um die Aktivitäten zu koordinieren
- Üblicherweise implementiert als Erweiterungen einer höheren Programmiersprache 

### Aufteilung der Arbeit (work partitioning)

- Identifizieren der Teilaufgaben, die parallel ausgeführt und auf die Prozessoren verteilt werden können
  - Das parallele Programm ist **konform** zur sequentiellen Semantik
  - Zwei Programmsegmente S1 und S2, die in einem sequentiellen Programm nacheinander ausgeführt werden, können parallel ausgeführt werden, *wenn S1 unabhängig von S2 ist*
- Thread/Prozess: Code , der auf einem Prozessor oder Prozessorkern eines Multiprozessors ausgeführt wird

#### Datenparallelismus (data-level parallelism)

- Berechnungen von verschiedenen Datenelementen sind unabhängig (Feld,

  Matrix)

  - Beispiel: Matrixmultiplikation

- Single Program Multiple Data (SPMD)

  - Eine Berechnung (eine Funktion) wird auf alle Datenelemente eines Feldes ausgeführt

  - Skalierung der Problemgröße

#### Funktionsparallelismus (function-level-parallelism, task-level parallelism)

- Unabhängige Funktionen werden auf *verschiedenen* Prozessoren ausgeführt

### Koordination (coordination)

- Parallel auf den verschiedenen Prozessoren laufende Threads müssen koordiniert werden, *so dass das Ergebnis dasselbe ist wie bei einem entsprechenden sequentiellen Programm*
- Synchronisation und Kommunikation
  - **Austausch von Informationen über gemeinsamem Speicher** oder **über explizite Nachrichten**
  - Zusätzlicher Zeitaufwand hat Auswirkung auf die Ausführungszeit des parallelen Programms

#### Gemeinsamer Speicher (Shared Memory)

- Kommunikation und Koordination von Prozessen (Threads) über gemeinsame Variablen und Zeiger, die gemeinsame Adressen referenzieren
- **Kommunikationsarchitektur**
  - Verwendung konventioneller Speicheroperationen für die Kommunikation über gemeinsame Adressen
  - **Atomare** Synchronisationsoperationen!

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-10%2022.30.42.png" alt="截屏2020-07-10 22.30.42" style="zoom:67%;" />

##### Shared-Memory-Programmiermodell: Primitive

| Name            | Syntax                           | Fumktion                                                     |
| --------------- | -------------------------------- | ------------------------------------------------------------ |
| ``CREATE``      | ``CREATE(p,proc,args)``          | Generiere Prozess, der die Ausführung bei der Prozedur `proc` mit den Argumenten `args` startet |
| ``G_MALLOC``    | ``G_MALLOC(size)``               | Allokation eines gemeinsamen Datenbereichs der Größe `size` Bytes |
| ``LOCK``        | ``LOCK(name)``                   | Fordere wechselseitigen exklusiven Zugriff an                |
| ``UNLOCK``      | ``UNLOCK(name)``                 | Fordere wechselseitigen exklusiven Zugriff an                |
| `BARRIER`       | `BARRIER(name,number)`           | Globale Synchronisation für `number` Prozesse                |
| `WAIT_FOR_END`  | `WAIT_FOR_END(number)`           | warten. bis `number` PRozesse terminieren                    |
| `WAIT_FOR_FLAG` | `while (!flag);` or `WAIT(flag)` | Warte auf gesetztes flag; entweder wiederholte Abfrage (spin) oder blockiere; |
| `SET FLAG`      | `flag=1;` or `SIGNAL(flag)`      | Setze flag; weckt Prozess auf, der `flag` wiederholt abfragt |

#### Nachrichtenorientiertes Programmiermodell (Message Passing)

- Kommunikation der Prozesse (Threads) mit Hilfe von Nachrichten
  - **KEIN** gemeinsamer Adressbereich

- **Kommunikationsarchitektur**

  - Verwendung von korrespondierenden Send- und Receive-Operationen

    - **Send**: Spezifikation eines lokalen Datenpuffers und eines Empfangsprozesses (auf einem entfernten Prozessor)

      ```c
      SEND(src_addr, size, dest, tag)
      ```

    - **Receive**: Spezifikation des Sende-Prozesses und eines lokalen Datenpuffers, in den die Daten ankommen

      ```C
      RECEIVE(buffer_addr, size,src,tag)
      ```

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-10%2022.35.07.png" alt="截屏2020-07-10 22.35.07" style="zoom:67%;" />

##### Message Passing: Primitive

| Name      | Syntax                               | Funktion                                                     |
| --------- | ------------------------------------ | ------------------------------------------------------------ |
| `CREATE`  | `CREATE(procedure)`                  | Erzeuge Prozess, der bei `procedure` startet                 |
| `SEND`    | `SEND(src_addr,size,d est,tag)`      | Sende `size` Bytes von Adresse `src_addr` an `dest` Prozess mit `tag` Identifier |
| `RECEIVE` | `RECEIVE(buffer_addr, size,src,tag)` | Empfange eine Nachricht mit der Kennung `tag` vom `src`-Prozess und lege `size` Bytes in Puffer bei `buffer_addr` ab |
| `BARRIER` | `BARRIER(name,number)`               | Globale Synchronisation von `number` Prozessen               |

