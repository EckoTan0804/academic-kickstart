---
# Title, summary, and position in the list
linktitle: "Grundlagen"
summary: ""
weight: 40

# Basic metadata
title: "Grundlagen"
date: 2020-07-31
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Vorlesung", "Zusammenfassung", "Rechnerstruktur"]
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
        parent: vektor-verarbeitung
        weight: 1

---

{{% alert note %}} 

TL;DR und Beispiele siehe [Übungszusammenfassung]({{< relref "../uebung-zusammenfassung/ub9-vektorrechner.md" >}})

{{% /alert %}}

## Einführung

Einfaches Bsp:

```
Y = a * X + Y
```

Wobei:

- `a`: eine Konstante
- `X`, `Y`: Vektoren

$\forall i, i = \text{Anzahl der Elemente des Vektors X und des Vektors Y}$

Es ergibt sich der neue Wert für `Y[i]` aus der Multiplikation von a mit `X[i]` und der Addition des Zwischenergebnisses mit `Y[i]`

In **MIPS-Notation**:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2022.26.15.png" alt="截屏2020-07-31 22.26.15" style="zoom:80%;" />

Analyse:

- Die Schleife wird ungefähr 600 Mal durchlaufen
- <span style="color:red">Hoher Aufwand</span>
  - In einem Schleifendurchlauf werden jeweils die Elemente von X und Y addiert und nach der Berechnung wird das Ergebnis gespeichert. Die Adressen werden aktualisiert und das Abbruchkriterium wird geprüft.🤪
  - Konflikte aufgrund von Datenabhängigkeiten
  - Multizyklus-Operationen

## Wie arbeiten Vektorprozessoren?

💡 Idee der Vektor-Prozessoren: **SIMD-Verarbeitung** (Single Instruction - Multiple Data)

- Verarbeitung von Vektoren in einem Rechenwerk mit *Pipeline-artig* aufgebauten Funktionseinheiten
- Bereitstellung von Vektoroperationen

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2022.32.49.png" alt="截屏2020-07-31 22.32.49" style="zoom:80%;" />

## Vektorprozessor

 **Vektorprozessor (Vektorrechner)** = ein Rechner mit pipelineartig aufgebautem/n Rechenwerk/en zur Verarbeitung von Arrays von Gleitpunktzahlen.

- **Vektor** = Array (Feld) von Gleitpunktzahlen
- **Vektoreinheit**: ein Satz von Vektorpipelines in Rechenwerk jedes Vektorrechners
- **Skalarverarbeitung**: Verknüpfung einzelner Operanden
- Ein Vektorrechner enthält neben der Vektoreinheit auch noch eine oder mehrere **Skalareinheiten**. Dort werden die skalaren Befehle ausgeführt (Befehle, die nicht auf ganze Vektoren angewendet werden sollen)

- Die Vektoreinheit und die Skalareinheit(en) können parallel zueinander arbeiten

  (d.h. Vektorbefehle und Skalarbefehle können parallel ausgeführt werden.)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2023.07.02.png" alt="截屏2020-07-31 23.07.02" style="zoom:80%;" />

- Die Pipeline-Verarbeitung wird mit einem Vektorbefehl für zwei Felder von Gleitpunktzahlen durchgeführt.
- Die bei den Gleitpunkteinheiten skalarer Prozessoren nötigen Adressrechnungen entfallen.

### Vektorbefehle

![截屏2020-08-01 12.44.22](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-01%2012.44.22.png)

![截屏2020-08-01 12.44.43](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-01%2012.44.43.png)

### Pipelining

- Taktdauer = Dauer der **längsten** Teilverarbeitungszeit zuzüglich der Stufentransferzeit

- Bsp: `B[i] + C[i]` mit $i = 1, 2, \dots, N$

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2023.11.42.png" alt="截屏2020-07-31 23.11.42" style="zoom:80%;" />

#### Verkettung

- Erweitert das Pipeline-Prinzip auf eine Folge von Vektoroperationen

- Die (spezialisierten) Pipelines werden miteinander **verkettet**

  (Die Ergebnisse einer Pipeline werden **SOFORT** der nächsten Pipeline zur Verfügung gestellt.)

- Bsp 1

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2023.20.45.png" alt="截屏2020-07-31 23.20.45" style="zoom:80%;" />

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-31%2023.21.38.png" alt="截屏2020-07-31 23.21.38" style="zoom:67%;" />


#### Typ von Pipeline

- **Multifunktions- oder spezialisierte Pipelines**

  Zur Realisierung der arithmetisch-logischen Verknüpfung von Vektoren

  - **Multifunktions-Pipelines**
    - erfordert eine höhere Stufenzahl, als sie zur Durchführung einer Verknüpfungsoperation notwendig wäre
    - Für die gerade aktuelle Operation werden alle nicht benötigten Stufen der Pipeline übersprungen.
  - **Spezialisierte Pipelines**
    - Durchführung von speziellen Funktionen
    - Relativ einfache Hardware und Steuerung
    - Man benötigt mehrere unabhängige Pipelines, um alle wichtigen Verknüpfungen durchführen zu können. 🤪

### Parallelarbeit in Vektorrechner

- **Vektor-Pipeline-Parallelität**

  durch die Stufenzahl der betrachteten Vektor-Pipeline gegeben

- Mehrere Vektor-Pipelines in einer Vektoreinheit

  - mehrere, meist funktional verschiedene Vektor-Pipelines in einer Vektoreinheit, durch Verkettung hintereinander geschaltet

- Vervielfachung der Pipelines

  - Bei Ausführung eines Vektorbefehl pro Takt nicht nur ein Paar von Operanden in eine Pipeline, sondern jeweils ein Operandenpaar in zwei oder mehr parallel arbeitende gleichartige Pipelines eingespeist werden.

- Mehrere Vektoreinheiten

  die parallel zueinander nach Art eines speichergekoppelten Multiprozessors arbeiten.

#### Parallelitätsebenen in der Software

- Vektor-Pipeline-Parallelität wird durch die **Vektorisierung der innersten Schleife** mittels eines vektorisierenden Compilers genutzt.

- Mehrere Vektor-Pipelines in einer Vektoreinheit können durch **Verkettung von Vektorbefehlen** oder durch **einen Vektor-Verbundbefehl** (beispielsweise Vector-Multiply-Add) genutzt werden.
- Bei Vervielfachung der Pipelines: Vektorisierung über die innerste Schleife
- Mehrere Vektoreinheiten: durch ähnliche Parallelisierungsmechanismen wie für speichergekoppelte Multiprozessoren genutzt

#### Speicherverschränkung (Memory Interleaving)

Technik, um die Zugriffsgeschwindigkeit auf den Hauptspeicher stärker an die Verarbeitungsgeschwindigkeit der CPU anzupassen.

- **Parallelisierung der Speicherzugriffe**
  - Speicher wird in $n$ **Speichermodule (oder Speicherbänke)** $M\_0,\dots,M\_{n-1}$ unterteilt und jede Speicherbank mit einer eigenen Adressierlogik versehen.

    $\to$ **$n$-fache Verschränkung**

  - Der Zugriff auf die Speicherplätze erfolgt zeitlich verschränkt.

- **Verschränkungsregel**

  - 💡Verteilung der Speicherplätze auf Speicherbänke Speicherplatz $A\_i$ wird in der Speicherbank $M\_j$ gespeichert, genau dann wenn
    $$
    j = i \bmod n
    $$

  - Zuteilung

    - der Adressen $A\_0, A\_n, A\_{2n},\dots$ auf die Speicherbank $M\_0$
    - der Adressen $A\_1, A\_{n+1}, A\_{2n+1},\dots$ auf die Speicherbank $M\_1$
    - etc.

  - $n$-fache Verschränkung: nach einer gewissen Anlaufzeit werden in jedem Speicherzyklus $n$ Speicherworte geliefert.
  
  - Bsp:
  
    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-06%2012.36.19.png" alt="截屏2020-08-06 12.36.19" style="zoom:67%;" />

