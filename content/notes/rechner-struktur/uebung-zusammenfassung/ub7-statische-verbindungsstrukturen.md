---
# Basic info
title: "Ub7-Statische Verbindungsstrukturen"
date: 2020-07-27
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Übung", "Zusammenfassung", "Rechnerstruktur"]
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
        parent: uebung-zusammenfassung
        weight: 14

weight: 171
---

- In statischen Netzen existieren **fest installierte** Verbindungen zwischen Paaren von Netzknoten

- Steuerung des Verbindungsaufbaus ist Teil der Knoten

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-27%2012.46.57.png" alt="截屏2020-07-27 12.46.57" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-27%2012.47.19.png" alt="截屏2020-07-27 12.47.19" style="zoom:80%;" />

##  $K$-ärer $n$-Kubus (Cubes, Würfel)

- $n$: Dimension

- $K$: Anzahl der Knoten, die einen Zyklus in einer Dimension bilden (Rückwärtskanten)

- $\Rightarrow$ Insgesamt $N = K^n$ Knoten

- Adressierung der Knoten: n-stellige k-äre Zahl der Form $a\_0, a\_1, \dots, a\_{n-1}$

  - Jede Stelle $0 \leq a\_i < K$ stellt die Position des Knotens in der entsprechenden $i$-ten Dimension dar, mit $0 \leq i \leq n-1$ 

  - Ein Nachbarknoten in der $i$-ten Dimension zu einem Knoten mit Adresse $a\_0, a\_1, \dots, a\_{n-1}$ kanan erreicht werden mit 
    $$
    a\_0, a\_1,\dots, a\_(i \pm 1) \bmod k, \dots, a\_{n-1}
    $$

- Knotengrad: $2n$
- Diameter: $n\left\lfloor\frac{k}{2}\right\rfloor$

### Bsp 1

- $\color{orange}{n=3}$

- $\color{green}{K=2}$

  ![截屏2020-07-27 12.53.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-27%2012.53.48.png)

- Adresse: $\color{orange}{3}$-stellige $\color{green}{2}$-näre Zahl $a\_0, a\_1, a\_2$
  - $a\_i \in (0, 2)$

### Bsp 3

- $\color{orange}{n=3}$

- $\color{green}{K=3}$

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-27%2012.57.58.png" alt="截屏2020-07-27 12.57.58" style="zoom:80%;" />

- Ein Nachbarknoten in der $i$-ten Dimension zu einem Knoten mit Adresse $a\_0, a\_1, \dots, a\_{n-1}$ kanan erreicht werden mit 
  $$
  a\_0, a\_1,\dots, a\_(i \pm 1) \bmod k, \dots, a\_{n-1}
  $$

  - Von 1**1**0 zu 1**0**0

    $a\_1 = 1 \Rightarrow (a\_1 - 1) \bmod 3 = 0$ 👏

  - Von **2**10 zu **0**10

    $a\_{0}=2 \Rightarrow\left(a\_{0}+1\right) \bmod 3=0$ 👏



## Ring (Aufg. 2)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-27%2013.03.39.png" alt="截屏2020-07-27 13.03.39" style="zoom:80%;" />

### (a) Charakterisierung

- Verbindungsgrad: 4

  > Jeder Knoten verbindet sich mit vier Nachbaren.

- Durchmesser: 2

  ![截屏2020-07-27 13.08.15](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-27%2013.08.15.png)

- min. Bisektionsbreite: 6

  > Die minimale Bisektionsbreite ist wie folgt definiert: 
  >
  > Schneidet man einen Graphen in zwei *gleich* große in sich zusammenhängende Teile und betrachtet die Menge der Kanten, die diesen Schnitt kreuzen, so bezeichnet man die Kardinalität der kleinsten Kantenmenge – über alle möglichen Schnitte – als **minimale Bisektionsbreite**.

  ![截屏2020-07-27 13.11.49](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-27%2013.11.49.png)

### (b) **Art des Verbindungsnetzwerkes** 

Chordaler Ring mit Knotengrad 4

![截屏2020-07-27 13.27.14](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-27%2013.27.14.png)

### (c) Redundanz

**Liegt Redundanz vor? Wenn ja, wieviele Verbindungsleitungen können ausfallen, bevoreine Verbindung zwischen zwei beliebigen Knoten nicht mehr geschalten werden kann?**

- Es liegt Redundanz vor.

- Verbindungsgrad jedes Knotens ist 4 und die bidirektionale Leitungen werden verwendet

  $\Rightarrow$ **Bis zu drei Leitungen** können sausfallen und dennoch jeder Knoten von einem anderen erreicht werden

  (Allerdings kann beim Ausfall einer Kante der Durchmesser steigen, das heißt es könnten längere Wege notwendig sein.)



## Zusammenfassung von Charakterisierung der Netzwerktopologien

- $N$: \#Knoten

|                       | Ring                                                         | 2D-Gitter  | (binärer) Baum                                        | (n-dim) Hyperkubus |      |
| --------------------- | ------------------------------------------------------------ | ---------- | ----------------------------------------------------- | ------------------ | ---- |
| Knotenzahl            | $N$                                                          | $N=n^2$    | $N$                                                   | $N=2^n$            |      |
| Verbingdungsgrad      | 2                                                            | 2 $\sim$ 4 | 1 $\sim$ 3                                            | $\log \_{2} N = n$ |      |
| Durchmesser           | <li> Unidirktional: $N-1$<br /><li>Bidirektional: $\frac{N}{2}$ | $2(n-1)$   | $2\left(\left\lceil\log \_{2} N\right\rceil-1\right)$ | $\log \_{2} N = n$ |      |
| min. Bisektionsbreite | 2                                                            | $n$        | 1                                                     | $2^{n-1} = N/2$    |      |

