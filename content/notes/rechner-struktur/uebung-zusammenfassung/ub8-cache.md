---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 180

# Basic metadata
title: "Ub8-Cache"
date: 2020-08-01
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Übung", "Zusammenfassung", "Rechnerstruktur"]
categories: ["Computer Structure"]
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
    rechner-struktur:
        parent: uebung-zusammenfassung
        weight: 15

---

## Motivation

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-05%2023.40.21.png" alt="截屏2020-08-05 23.40.21" style="zoom:67%;" />

- Ausnutzung der räumlichen und zeitlichen Lokalität von Anwendungen

- 90/10-Regel

  > Die 90/10 Regel besagt, dass **bei 90 % der Speicherzugriffe innerhalb eines Programms nur etwa10% der Daten benötigt werden**, die das Programm insgesamt während seiner Laufzeit benötigt. Somit ist also ein deutlich kleinerer Speicher ausreichend, um die meisten Anfragen beantworten zu können.

## Parameter

### Organisation

- Direkt abgebildet (direct mapped) 
- Satzassoziativ (set associative) 
- Vollassoziativ (fully associative)

### Hierarchie

- Mehrere Level (L1, L2, L3)

- im Multiprozessorfall: private vs. shared 
- inclusive vs. exclusive

### Größe

- Cacheline-Size

- (\# Cachelines bzw. Assoziativität) * \# Sätzen

### Ersetzungsstrategie

- least recently used (LRU) 
- least frequently used (LFU) 
- Round Robin

- Random

### Schreibstrategie

- write-through vs. write-back 
- write-allocation vs. no write-allocation

### Cachekohärenzprotokoll (im Multiprozessorfall)

- Busbasiert 
- Verzeichnisbasiert

## **Eigenschaften**

Bei Caches kann zwischen unterschiedlichen auftretenden Fehlzugriffen unterschieden werden

- **Compulsory Miss**
  - Miss, der durch *ersten* Zugriff auf ein Datum entsteht
  - Daten also *noch nicht im Cache* und daher Miss *nicht vermeidbar*
- **Capacity Miss**
  - Kapazität des Caches *zu klein*, weshalb bereits gecachtes Datum aus Cache entfernt werden musste

- **Conflict Miss**
  - *Set zu klein* und bereits gecachtes Datum wurde aus Set entfernt (Cache nicht zwangsläufig voll)
  - Nur bei *nicht-vollassoziativem* Cache
- **Coherency Miss**
  - Nur bei Multiprozessor-Systemen mit Kohärenzprotokoll
  - Unterscheidung zwischen False- und True-Sharing 
  - False-Sharing, falls nicht eigentliches Wort sondern anderes Wort in Cache-Block geändert wurde

## Cacheleistung

- Gegeben
  - Hit-Rate: $r\_H$
  - Miss-Rate: $r\_M = 1 - r\_H$
  - Zugriffszeit bei Hit: $t\_H$
  - Zugriffszeit Hauptspeicher (fall Miss): $t\_{Mem}$

- **Mittlere Zugriffszeit**
  $$
  t\_{a}=\underbrace{r\_{H} * t\_{H}}\_{\text {Hit }}+\underbrace{r\_{M} * t\_{M e m}}\_{\text {Miss }}
  $$

- **Mehrstufige Hierarchie: Konkretisierung des Miss-Zweigs**
  $$
  t\_{a}=\underbrace{r\_{H 1} * t\_{L 1}}\_{\text {Hit L1 }}+\underbrace{r\_{M 1} *(\underbrace{r\_{H 2} * t\_{L 2}}_{\text {Hit L2 }}+\underbrace{r\_{M 2} * t\_{M e m}}\_{\text {Miss L2 }})}\_{\text {Miss L1 }}
  $$

