---
# Basic info
title: "Multiprozessoren mit verteiltem Speicher"
date: 2020-07-29
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
        weight: 8

# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 37
---

## **Allgemeine Rechnerorganisation**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2012.24.23.png" alt="截屏2020-07-31 12.24.23" style="zoom:80%;" />

## **Nachrichtenorientiertes Programmiermodell (Message Passing Model)**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2012.25.08.png" alt="截屏2020-07-31 12.25.08" style="zoom: 67%;" />

### **Message-passing Primitive**

#### Synchrones Message-Passing

- Sender blockiert, bis die Nachricht beim Empfänger angekommen ist. 

- Ebenso muss der Empfänger blockieren, bis die Nachricht empfangen worden ist und in den gewünschten lokalen Speicher kopiert worden ist

- Bsp

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2012.27.23.png" alt="截屏2020-07-31 12.27.23" style="zoom:80%;" />

  - A=10 wird an den Empfänger P2 geschickt
  - P2 muss blockieren, bis der Inhalt der Nachricht (10) in die lokale Variable B kopiert worden ist.

- <span style="color:red">Deadlock-Gefahr!!!!</span>

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2012.57.41.png" alt="截屏2020-07-31 12.57.41" style="zoom:80%;" />

  - Der Sender und der Empfänger blockieren, bis die Nachrichtenübertragung beendet ist, d.h. bis ein korrespondierendes `SEND`/`RECV` Paar ausgeführt ist
  - P1 und P2 sind blockiert bei ihren jeweiligen `SEND`-Primitiven, da beide auf ihr korrespondierendes `RECV` warten. 🤪

- Kombination von Synchronisation und Kommunikation in einer Primitive kann zu **<span style="color:red">L eistungsverlust</span>** führen!

#### Asynchrones Message-Passing

💡 Überlappung von Kommunikation und Berechnung

- Blockierendes asynchrones `SEND`
  - Gibt die Kontrolle an den Sende-Prozess zurück, wenn 
    - die zu versendenden Daten in einem Puffer kopiert worden sind und 
    - nicht mehr durch die Berechnung verändert werden können
  - Blockierendes asynchrones `RECV`
    - Gibt ebenso die Kontrolle NICHT an den Empfangsprozess zurück, bis die Nachricht in einen lokalen Adressraum gespeichert worden ist
  - Nichtblockierende asynchrones Kommunikationsprimitive
    - Die Kontrolle wird **SOFORT** an den Sende- bzw. Empfangsprozess zurückgegeben und der Datentransfer läuft im Hintergrund
  - Probe-Funktionen
    - Prüfen ob Daten vom lokalen Adressraum des Senders in einen Puffer kopiert worden sind, bzw. ob der Empfänger die Daten kopiert hat



### **Message-passing Protokolle**

#### Sender-initiiertes Message-Passing Protokoll (synchron): Drei-Phasen- Protokoll

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2013.07.50.png" alt="截屏2020-07-31 13.07.50"  />

#### Empfänger-initiiertes Message-Passing Protokoll (synchron)

![截屏2020-07-31 13.08.11](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2013.08.11.png)

### **Hardware-Unterstützung für Message-passing Protokolle**

#### Software-Overhead

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2013.09.20.png" alt="截屏2020-07-31 13.09.20" style="zoom:70%;" />

#### Hardware-Support für Message-Passing: DMA

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2013.09.56.png" alt="截屏2020-07-31 13.09.56" style="zoom:80%;" />

#### Hardware-Support für Message-Passing: Kommunikationsprozessor

![截屏2020-07-31 13.10.19](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2013.10.19.png)