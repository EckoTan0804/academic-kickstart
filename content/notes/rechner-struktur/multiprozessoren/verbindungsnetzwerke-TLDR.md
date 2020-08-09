---
# Basic info
title: "Verbindungsnetzwerke (TL;DR)"
date: 2020-07-24
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
        weight: 5

# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 34
---

## Definition

### Latenz (latency)

- Übertragungszeit einer Nachricht

- Software overhead
- Kanalverzögerung (channel delay)
- Schaltverzögerung, Routing-Verzögerung (switching delay, routing delay)
- Blockierungszeit (contention time)
- Blockierung (contention)

### Durchsatz oder Übertragungsbandbreite

*Maximale* Übertragungsleistung des Verbindungsnetzwerkes oder einzelner Verbindungen (Mbit/s oder MB/s)

### Diameter oder Durchmesser $r$ (diameter)

*Maximale Distanz* für die Kommunikation zweier Prozessoren

### Verbindungsgrad eines Knotens $P$ (node degree, connectivity)

Anzahl der direkten Verbindungen, die von einem Knoten zu anderen Knoten bestehen.

### Mittlere Distanz $d\_a$ (average distance) zwischen zwei Knoten

Anzahl der Links auf dem kürzesten Pfad zwischen zwei Knoten

### Komplexität oder Kosten

### Erweiterbarkeit

### Skalierbarkeit

### Ausfallstoleranz oder Redundanz

### Verbindungsnetzwerk (IN)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2020.36.03.png" alt="截屏2020-07-24 20.36.03" style="zoom:100%;" />

- Knoten
- Schaltelement (Switch)
- Link
- Nachricht (message)
  - Unicast
  - Multicast
  - Broadcast
- Paket fester Länge: Auftrennung der Nachricht
  - Envelope
    - Header
    - Errorcode
    - Trailer
  - Payload

### Switching strategy

- Circuit switching

- Packet switching

  - store-and-forward

  - cut-through

## Latenz- und Bandbreitenmodell: End-to-end packet latency model

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2022.02.34-20200806221332097.png">}}

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2022.04.04.png" alt="截屏2020-07-24 22.04.04" style="zoom:80%;" />

**End-to-end packet latency = Sender OH + Time of flight (incl. switching time) + Transmission time + Routing time + Receiver OH**

Annahmen:

- Ein Paket hat auf dem Weg von der Quelle zum Ziel $L$ Schaltelemente zu passieren
- Paket umfasst $N$ Phits
- Die Routing-Entscheidung in einem Schaltelement benötigt $R$ Netzwerkzyklen
- Sender OH und Receiver OH sind fest

|                                                              | Time of Flight<br />(in Zyklus) | Transmission time<br />(in Zyklus) | Routing time<br />(in Zyklus)                                |
| ------------------------------------------------------------ | ------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| **Circuit Switching**                                        | $L$                             | $N$                                | <li>Informieren von Aufbau des Wegs zwischen Quelle und Ziel: $L$<li>Routing-Entscheidung (Routing OH) in Schaltelemente: $L \times R$<br />​<br />Insgesamt: $L + L \times R$ |
| **Packet Switching (Store-and-forward)<br />**<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2011.26.57.png" alt="截屏2020-07-25 11.26.57" style="zoom:80%;" /> | $L \times N$                    | $N$                                | $L \times R$                                                 |
| **Packet Switching (cut-through)** <br /><img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2011.54.56.png" alt="截屏2020-07-25 11.54.56" style="zoom:80%;" /> | $L$                             | $N$                                | $L \times R$                                                 |



## Topologie

### Statisch

- Vollständige Verbindung
- Gitter
  - 1-dim (lineares Feld, Kette)
  - k-dimensionales Gitter mit N Knoten 
- Ring
  - Unidirektional
  - Bidirektional
  - Chordaler
- Baum (Tree)
  - Fat tree
- Kubus
  -  K-ärer n-Kubus (Cubes, Würfel)
  - Hyperkubus (Hypercubes)

### Dynamisch

- Bus
  - Split-phase Busprotokollen
- Kreuzschiene (Crossbar)
  - Schalterelemente (2x2 Kreuzschienenverteiler)
- Schalternetzwerk
  - Permutationsnetze
    - Einstufige
    - Mehrstufige
    - reguläre
    - Irreguläre
  - Permutation
    - Mischpermutation $M$ (Perfect Shuffle)
    - Kreuzpermutation $K$ (Butterfly)
    - Tauschpermutation $T$ (Butterfly)