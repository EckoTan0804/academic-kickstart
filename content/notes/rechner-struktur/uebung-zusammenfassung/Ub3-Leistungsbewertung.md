---
# Basic info
title: "Ub3-Leistungsbewertung"
date: 2020-07-08
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
        weight: 5
---

## TL;DR

- CPI: $CPI = \frac{c}{i}$
  - $CPI = \displaystyle \sum_{i=1}^{n}CPI_i \cdot Anteil_i$
- MIPS: $MIPS = \frac{i}{t\cdot 10^6}$
- Taktrate: $f=\frac{c}{t}$
- CPU-Zeit: $t_{cpu}=c \cdot t_{Zyklus}$

- Anzahl Instruktionen: $i = \sum i_{typ}$
- Takyzyklen: $c=\sum i_{typ}\cdot c_{typ}$
- Ausführungszeit: $t_{exec}=c \cdot t_{cyc}$
- Bedienzeit: $t_{Bedien} = t_{Zugriff} + t_{Übertragung}$
- Maximaler Durchsatz: $D_{\max} = \frac{1}{t_{Bedien}}$
- Auslastung: $U = \frac{D}{D_{\max}}$
- Gesetz von Little: 
  - $Q = W \cdot D$ (#Aufträge in der Warteschlange = Wartezeit $\cdot$ Durchsatz)
  - $k = \lambda \cdot t$ (#Aufträge = Durchsatz $\cdot$ Antwortzeit)
- Reaktionzeit: $\text{Reaktionszeit} = \text{Wartezeit} + \text{Bedienzeit}$

## Was ist Leistung?

- **Anwendersicht**: Reduzierung von 

  - Antwortzeit (response time)
    - Latenzzeit

    - CPU Time (User, System)
  - Ausführungszeit (execution time)

- **Betriebssicht**: Erhöhung von Anzahl 

  - durchgefu ̈hrter Jobs

  - Durchsatz

  - Energieeffizienz (Betriebskosten)



## Leistungsbewertung

### Auswertung von Hardwaremaßen und Parametern

- **Prozessortakt**: gibt lediglich den Arbeitstakt (min/typ/max) des Prozessors an.
  
- KEIN Maß für Leistungsfähigkeit (da keine Aussage u ̈ber Effizienz, Gu ̈te des Befehlssatzes etc.)
  
- **CPI**

  - **C**ycles **P**er **I**nstruction (Zyklen pro Instruktion)
    $$
    C P I=\frac{c}{i}
    $$

    - $c$: #alle Zyklen
    - $i$: #Instruktion
      - ist bedingt durch die Befehlssatzarchitektur und die Güte des Compilers

  - ein Maß für die *Effizienz* einer Architektur

  - werden durch die Organisation und die Befehlssatzarchitektur beeinflusst

  - Zur Leistungsbewertung als **alleinige** Maßzahl NICHT ausreichend: 

    Effizienz $\neq$ Geschwindigkeit

- **MIPS** 👍

  - **M**illion **I**nstructions **p**er **S**econd
    $$
    M I P S=\frac{i}{t * 10^{6}}=\frac{f}{CPI * 10^{6}}
    $$

  - ideal, weil zwei Maßzahlen (Takt, CPI) zusammengeführt werden.

  - Nur unter gleichen Bedingungen (Sourcecode, Compiler, OS) direkt vergleichbar.

- **MFLOPS** 

  - Millions of Floatingpointoperations Per Second

  - Wie MIPS, wobei Anzahl der Befehle und Ausführungszeit **nur für Fließkommaberechnung**
    $$
    \text{MFLOPS} =\frac{\text { Anzahl der ausgefuhrten Gleitkommaoperationen }}{10^{6} \times \text { Ausführungszeit }}
    $$
    

Zur Berechnung benötigte Maße:

- **Taktrate (Frequenz)**
  $$
  f=\frac{c}{t}=\frac{i * CPI}{t}[\mathrm{Hz}]
  $$

- **CPU-Zeit** (Ausführungszeit)
  $$
  t_{c p u}=c * t_{Z y k l u s}
  $$
  - $c$: #alle Zyklen

  - $t_{Zyklus}$: Zykluszeit
    $$
    t = \frac{1}{f}
    $$

    - hängt von der Organisation und der Technologie ab

- **Speedup**
  $$
  \text{Speedup} = \frac{t_{cpu, old}}{t_{cpu, new}} = \frac{CPI_{old}}{CPI_{new}}
  $$

### Benchmark

$$
\text {SPECratio}=\frac{\text {Referenzzeit}\_{x}}{\text {Laufzeit}\_{x} \text { auf Testsystem}}
$$

Bsp: Aufg. 5 (b)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-20%2016.20.14.png" alt="截屏2020-06-20 16.20.14" style="zoom: 67%;" />

$\Rightarrow \text{Referenzzeit}_{462.libquantum} = 613 \cdot 33.8s = 20719.4s$

### Modelltheoretische Verfahren

- **Bedienzeit**:

$$
t_{\text{Bedien}} = t_{\text{Zugriff}} + t_{\text{Übertragung}}
$$

> Bsp: Aufg.6 (a)
>
> Zugriffszeit 12 ms, Datenrate 6 MByte/s, Schreib-/Leseauftrag 100kB
> $$
> \begin{aligned}
> t_{\text{Bedien}} &= t_{\text{Zugriff}} + t_{\text{Übertragung}} \\
> &= 12ms + \frac{100kB}{6\cdot 10^3 kB/s} \\
> &= 28.67ms
> \end{aligned}
> $$

- **Durchsatz**
  
  - Maximaler Durchsatz
  
$$
  \qquad D_{\max} = \frac{1}{t_{\text{Bedien}}}
$$
  
  - Nur Platten mit $D_{\max}$ > Ankunftsrate können eingesetzt werden, da sonst die Festplatte nicht genügend Zeit hat, um alle Aufträge rechtzeitig zu bedienen.
  
- **Auslastung**
  $$
  U = \frac{D}{D_{\text{max}}} = D \cdot t_{\text{Bedien}}
  $$

- **Warteschlange**
  - **Wartezeit**

    Gesetz von Little:
    $$
    Q = W \cdot D
    $$

    - $Q$: Anzahl von Aufträgen in der Warteschlange

    - $W$: Wartezeit
    - $D$: Durchsatz

    > Bsp: Aufg. 6 (d)
    >
    > drei Aufträge, Durchsatz=Ankunftsrate=40/s
    >
    > Wartezeit: $W = \frac{3}{40/s}=75ms$

  - **Reaktionszeit**
    $$
    \text{Reaktionszeit} = \text{Wartezeit} + \text{Bedienzeit}
    $$
    

