---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 240

# Basic metadata
title: "Cache Aufgaben"
date: 2020-08-06
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Alte Klausur", "Zusammenfassung", "Rechnerstruktur"]
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
        parent: alte-klausur-zusammenfassung
        weight: 40

---

## Cache Miss

- Compulsory/Cold Miss
- Capacity Miss
- Conflict Miss
- Coherency Miss
  - Nur bei Multiprozessor-Systemen mit Kohärenzprotokoll
  - Unterscheidung zwischen False- und True-Sharing
  - False-Sharing, falls nicht eigentliches Wort sondern anderes Wort in Cache-Block geändert wurde (See also: [What’s false sharing and how to solve it](https://medium.com/@genchilu/whats-false-sharing-and-how-to-solve-it-using-golang-as-example-ef978a305e10))

### Bsp: Klausur SS17, Aufg. 3, (e)

- Multiprozessor mit gemeinsamen Speicher, der aus drei Prozessoreinheiten sowie Caches besteht.
- Jeder Cache bietet Platz für eine Cache-Zeile. 
- Cache-Protokoll: MESI Protokoll
- Zugriff auf vier Variablen (A, B, C, D), wobei
  - Variablen A,B und C dem selben Speicherblock angehören, also zusammen geladen werden
  - D einem anderen Speicherblock angehör
  - Beide Speicherblöcke werden auf die selbe Cache-Zeile abgebildet.

Klassifizieren Sie die auftretenden Cache Misses. Unterscheiden Sie dabei zwis- 4P chen True- und False-Sharing Misses.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-06%2023.40.02.png" alt="截屏2020-08-06 23.40.02" style="zoom:80%;" />

> - False-Sharing Miss in Zeile 6:
>
>   Variablen A, B, C angehören dem selben Speicherblock. Prozessor 1 hat in Zeile 4 Variable A verändert.
>
>   Laut MESI Protokoll muss der selbe Speicherblock in andere Cache des anderen Prozessors invalidiert werden. $\Rightarrow$ Cache Miss
>
>   Da NICHT eigentliches Wort B, sondern Variable A, also das andere Wort, in Cache-Block geändert wird, ist dieses Cache Miss daher ein False-Sharing Miss.
>
> - True-Sharing Miss in Zeile 9:
>
>   Variable B ist in Zeile 7 geändert. D.h. das eigentliche (dasselbe) Wort wird geändert
>
>   $\Rightarrow$ True-Sharing