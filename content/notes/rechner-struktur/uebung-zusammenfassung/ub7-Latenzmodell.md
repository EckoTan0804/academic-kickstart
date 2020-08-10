---
# Basic info
title: "Ub7-Latenzmodell"
date: 2020-07-26
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
        weight: 13

weight: 170
---

{{% alert warning %}} 

Achtung:

Die in Übung vorkommende Berechnung des Latenzmodells ist teilweise **inkonsistent** mit dem Modell, dem in Vorlesungsfolien auftritt. 😭

Der Übungsleiter schlägt vor, die alte Klausuren zu schauen und zu orientieren. 🤪

{{% /alert %}}

## End-to-end packet latency model

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2022.04.04.png" alt="截屏2020-07-24 22.04.04" style="zoom:80%;" />

**End-to-end latency**: Zeit, die benötigt wird von diesem Zeitpunkt an bis das gesamte Paket über das Verbindungsnetzwerk übertragen worden ist und in einem Puffer am Empfängerknoten abgelegt ist
$$
\text{End-to-end packet latency} = \text{sender OH} + \text{time of flight} + \text{transmission time} + \text{routing time} + \text{receiver OH}
$$


- **Sender overhead (OH)**

  Zusammenstellen des Pakets und Ablegen in Sendepuffer der Netzwerkschnittstelle

- **Time of flight**

  Zeit, um ein Bit von der Quelle zum Ziel zu senden, wenn Weg festgelegt und konfliktfrei

- **Switching time**

  Hängt von der Switching-Strategie ab

- **Transmission time**

  Zusätzliche Zeit, die benötigt wird, alle Bits eines Pakets zu übertragen, nachdem erstes Bit beim Empfänger angekommen ist

- **Routing time**

  Zeit, um den Weg aufzusetzen, bevor ein Teil des Pakets übertragen werden kann

- **Receiver overhead**

  Ablegen der Verwaltungsinformation und Weiterleiten des Pakts aus dem Empfangspuffer

## Switching Strategy

- Bestimmt, wie ein Weg in einem Verbindungsnetzwerk aufgebaut und ein Paket von der Quelle zum Ziel übertragen wird
- Modellannahmen
  - Pfad als Übertragungspipeline
  - Paket umfasst $N$ Phits und überquert $L$ Schaltelemente
    - Ein Phits enthält mehere Bits
    - Pro Zyklus wird 1 Phit übertragen

### **circuit switching**

- 💡 **Aufbau Weg zwischen Quelle und Ziel**, danach Ubertragung (Pipelining)

- Routing time

  - Zeit, die notwendig ist, ein Phit von der Quelle zum Ziel und

    zurück zu senden, um die Quelle zu informieren, dass der Weg aufgebaut ist

    (benötigt $L$ Zyklen, also 1 mal Time of Flight)

  - Routing-Entscheidung (Routing Overhead) in einem Schaltelement benötigt $R$ Netzwerkzyklen

- End-to-end packet latency (in Zyklus) ist: 

$$
\text{End-to-end packet latency} = \text{sender OH} + \underbrace{L}\_{\text{time of flight}} + \overbrace{N}^{\text{transmission time}} + \underbrace{L(R+1)}\_{\text{routing time}} + \text{receiver OH}
$$

### **Packet switching store-and-forward Modus**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2011.26.57.png" alt="截屏2020-07-25 11.26.57" style="zoom:80%;" />

- Pfad wird bestimmt bei Erreichen eines Schaltelements

- **Alle** Teile müssen angekommen sein, bevor ein Teil weitergeleitet wird
- Time of flight:
  - umfasst Zeit für Ubertragung eines Bits von Quelle zu Ziel (ohne Routing OH)
  - Hängt von Paketgröße ab: Paket $N$ Zyklen in Schaltelement

- End-to-end packet latency (in Zyklus) ist: 

$$
\text{End-to-end packet latency} = \text{sender OH} + \underbrace{L \times N}\_{\text{time of flight}} + \overbrace{N}^{\text{transmission time}} + \underbrace{L \times R}\_{\text{routing time}} + \text{receiver OH}
$$

### **Packet switching cut-through switching Modus**

- End-to-end packet latency (in Zyklus) ist: 

$$
\text{End-to-end packet latency} = \text{sender OH} + \underbrace{L }\_{\text{time of flight}} + \overbrace{N}^{\text{transmission time}} + \underbrace{L \times R}\_{\text{routing time}} + \text{receiver OH}
$$

### 

## Aufgabe

- $4 \times 4$ Mesh Verbindungsnetzwerk
- Ein Paket mit der Größe 100 Bytes soll vom linken oberen Knoten des Netzwerks zum rechten unteren Knoten übertragen werden
- Größe eines Phits: 10 Bits
- Frequenz des Verbindungsnetzwerks: 100 MHz
- Routing OH: 1 Taktzyklus
- Sender/Receiver OH: 10ns

{{% alert note %}} 

- 1 MHz = $10^3$ KHz = $10^6$ Hz
- 1 s = $10^3$ ms = $10^6$ $\mu$s = $10^9$ ns

{{% /alert %}}



### Teilaufg. (a)

**Berechnen Sie die end-to-end Latenz, falls als Switching-Strategie *circuit switching* verwendet wird.**

Lösung:

- Anzahl benötige Schaltelemente

  links oben zu recht unten

  $\Rightarrow$ 3 Schaltelemente nach unten 3 Schaltelemente nach rechts, insgesamt 6

- Zykluszeit: 
  $$
  T = \frac{1}{f} = \frac{1}{100 \cdot 10^6 Hz} = 10 \text{ ns}
  $$

- Time of Flight: KEINE switching OH
  $$
  \begin{aligned}
  \text{ToF} &= \text{#Schaltelemente} \cdot \text{Zykluszeit}  \\\\
  &= 6 \cdot 10 \text{ ns} = 60 \text{ ns}
  \end{aligned}
  $$

- Anzahl von Phits
  $$
  \begin{aligned}
  \text{#Phits} &= \frac{\text{Paketgröße}}{\text{Größe eines Phits}} \\\\
  &= \frac{100 \cdot 8 \text{ Bits}}{10 \text{ Bits}} = 80
  \end{aligned}
  $$

- Transmission time
  $$
  \text{Transmission time} = \text{#(noch zu übertragende) Phits} \cdot \text{Zykluszeit}
  $$
  {{% alert note %}} 

  ‼️ Während Time of Flight ist der erste Bits (also der erste Phits) von der Quelle zum Ziel gesendet. Für die Transmission Time heißt es dann zusätzliche Zeit für das Senden der restlichen Bits/Phits, also $80 - 1= 79$ Phits.

  {{% /alert %}}

  Daher:
  $$
  \text{Transmission time } = 79 \text{ Phits} \cdot 10 \text{ ns} = 790 \text{ ns}
  $$

- Routing time
  $$
  \begin{aligned}
  \text{Routing time } &= \text{#Schaltelemente} \cdot \text{Routing OH} + \text{Time of Flight} \\\\
  &= 6 \cdot 10 \text{ ns} + 60 \text{ ns} = 120 \text{ ns}
  \end{aligned}
  $$

Insgesamt:
$$
\begin{aligned}
\text{End-to-end Latenz } &= \text{sender OH} + \text{time of flight} + \text{transmission time} + \text{routing time} + \text{receiver OH} \\\\
& = (10 + 60 + 790 + 120 + 10) \text{ ns} = 990 \text{ ns}
\end{aligned}
$$

### Teilaufgabe (b)

**Berechnen Sie die end-to-end Latenz, falls als Switching-Strategie *packet switching im store-and-forward Modus* verwendet wird.**

Bei **packet switching im store-and-forward Modus**

- Jedes Schaltelement speichert **komplettes** Paket bevor es weitergeleitet wird

- Route wird nicht vorher festgelegt

  $\Rightarrow$ Time of flight abhängig von Paketgröße

Lösung:

- Time of Flight:

  > Die Time of Flight gibt die Zeit an bis das erste Phit am Ziel angekommen ist. Für das letzte Schaltelement müssen daher nicht erst alle Phits des Pakets übertragen werden, sondern es reicht ein zusätzlicher Zyklus. 
  >
  > Daher sollen $6 - 1 = 5$ Schaltelemente betrachtet werden.

  $$
  \begin{aligned}
  \text{Time of flight } &= \text{#überquerte Schaltelemente} \cdot \text{#Phits} \cdot \text{Zykluszeit} + \text{Zyklus für erstes Bit/Phit zum Ziel} \\\\
  &= (\text{#benötige Schaltelemente}-1) \cdot \text{#Phits} \cdot \text{Zykluszeit} + \text{Zyklus für erstes Bit/Phit zum Ziel} \\\\
  &= (6-1) \cdot 80 \cdot 10 \text{ ns} + 10 \text{ ns} = 4010 \text{ ns}
  \end{aligned}
  $$

- Routing time
  $$
  \begin{aligned}
  \text{Routing time } &= \text{#Schaltelemente} \cdot \text{Routing OH} \\\\
  &= 6 \cdot 10 \text{ ns}= 60 \text{ ns}
  \end{aligned}
  $$

Insgesamt:
$$
\begin{aligned}
\text{End-to-end Latenz } &= \text{sender OH} + \text{time of flight} + \text{transmission time} + \text{routing time} + \text{receiver OH} \\\\
& = (10 + 4010 + 790 + 60 + 10) \text{ ns} = 4880 \text{ ns}
\end{aligned}
$$

### 