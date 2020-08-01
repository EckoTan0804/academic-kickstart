---
# Title, summary, and position in the list
linktitle: "Eigenschaften"
summary: ""
weight: 41

# Basic metadata
title: "Eigenschaften"
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
        weight: 2

---

## Vektor Stride

<span style="color:red">Problem: die Elemente eines Vektors liegen NICHT in aufeinander folgenden Speicherzellen</span>

- Bsp: Matrix-Multiplikation (k=3)

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-01%2011.18.53.png" alt="截屏2020-08-01 11.18.53" style="zoom:80%;" />

**Vektor Stride** = Abstand zwischen Elementen, die in einem Register abgelegt werden müssen

Vektorprozessoren mit Vektorregister können *Strides größer 1* verarbeiten: 

- Vektorlade- und Vektrospeicherbefehle mit „Stride-Capability“

- Zugriff auf nicht sequentielle Speicherzellen und Umformen in dichte Struktur
- 🔴 <span style="color:red">Problem: Stride-Wert ist erst zur Laufzeit bekannt oder kann sich ändern</span>
- 🔧 Lösung
  - Ablegen des Stride-Wertes in ein *Allzweckregister* 
  - Vektorspeicherzugriffsbefehle greifen auf den Wert zu



## Bedingt ausgeführte Anweisungen

<span style="color:red">Problem: Programme mit if-Anweisungen in Schleifen können NICHT vektorisiert werden </span> $\to$ **<span style="color:red">Kontrollflussabhängigkeiten</span>**

Bsp:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-01%2011.26.10.png" alt="截屏2020-08-01 11.26.10" style="zoom:80%;" />

🔧 Lösung

- Bedingt ausgeführte Anweisungen

- Umwandlung von Kontrollflussabhängigkeiten in Datenabhängigkeiten

**Vektor-Maskierungssteuerung**

- **verwendet einen Boole‘schen Vektor der Länge der festgelegten MVL (maximale Vektorlänge)**, um die Ausführung eines Vektorbefehls zu steuern

  (in ähnlicher Weise wie bedingt ausgeführte Befehle eine Boole‘sche Bedingung verwenden, um zu bestimmen, ob eine Instruktion ein gültiges Ergebnis liefert oder nicht)

- Vektor-Mask-Register

  - Jede ausgeführte Vektorinstruktion arbeitet nur auf den Vektorelementen, deren Einträge eine **1** haben. 
  - Die Einträge im Zielvektorregister, die eine **0** im entsprechenden Feld des VM Registers haben, werden *nicht verändert*

- Bsp

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-01%2011.32.38.png" alt="截屏2020-08-01 11.32.38" style="zoom:80%;" />



## Dünn besetzte Matrizen

💡 Elemente eines Vektors werden in einer **komprimierten Form** im Speicher abgelegt

Bsp

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-01%2011.34.37.png" alt="截屏2020-08-01 11.34.37" style="zoom:80%;" />

- Summe der dünn besetzten Felder A und C mit Hilfe der Indexvektoren K und M. 
  - K und M zeigen jeweils die Elemente von A und C an, die **nicht 0** sind.

- Alternative Darstellung: Verwendung von Bit-Vektoren

### SCATTER-GATHER Operationen mit Index-Vektoren

unterstützen den Transport zwischen der *gepackten Darstellung* und der *normalen Darstellung dünn-besetzter Matrizen*

- `GATHER`-Operation

  - verwendet Index-Vektor und holt den Vektor, dessen Elemente an den Adressen liegen, die durch Addition einer Basisadresse und den Offsets im Index-Register berechnet werden
  - Nicht gepackte Darstellung im Vektorregister

- `SCATTER`-Operation: Speichern der gepackten Darstellung

- Bsp (entsprechend dem obigen Bild)

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-01%2011.40.51.png" alt="截屏2020-08-01 11.40.51" style="zoom:80%;" />

- <span style="color:red">Problem für vektorisierenden Compiler: konservative Annahmen wegen Speicherreferenzen</span>
  - Verwendung einer Software-Hash Tabelle
  - erkennt, wenn zwei Elemente innerhalb einer Iteration auf dieselbe Adresse zeigen