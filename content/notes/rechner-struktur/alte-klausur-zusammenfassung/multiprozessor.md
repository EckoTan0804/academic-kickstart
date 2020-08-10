---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 241

# Basic metadata
title: "Multiprozessor Aufgaben"
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
        weight: 41
---

## Verbindungsnetzwerk

### End-to-end latency model

Annahmen:

- Ein Paket hat auf dem Weg von der Quelle zum Ziel $L$ Schaltelemente zu passieren
- Paket umfasst $N$ Phits
- Die Routing-Entscheidung ($R\_{OH}$) in einem Schaltelement benötigt $R$ Netzwerkzyklen
- Sender OH und Receiver OH sind fest

**End-to-end packet latency = Sender OH + Time of flight (incl. switching time) + Transmission time + Routing time + Receiver OH**

- Sender OH: fest gegeben

- Time of Flight: 

  - Circuit Switching: $L \cdot T\_{\text{cycle}}$
  - Packet Switching
    - Store-and-forward: $L \cdot N \cdot T\_{\text{cycle}}$
    - Cut-through: $L \cdot T\_{\text{cycle}}$

- Transmission time: $(N-1)\cdot T\_{\text{cycle}}$

  > Das 1.Phit ist während Time-of-Flight schon gesendet. Daher gibt es noch restlich $N-1$ zu sendende Phits.

- Routing time:

  - Circuit switching: $\text{Time of Flight} + L \cdot R\_{\text{OH}}$
  - Packet Switching
    - Store-and-forward: $L \cdot R\_{\text{OH}}$
    - Cut-through: $L \cdot R\_{\text{OH}}$

- Receiver OH: fest gegeben



|                                                              | Time of Flight<br />(in Zyklus) | Transmission time<br />(in Zyklus) | Routing time<br />(in Zyklus)                                |
| ------------------------------------------------------------ | ------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| **Circuit Switching**                                        | $L$                             | $N-1$                              | <li>Informieren von Aufbau des Wegs zwischen Quelle und Ziel: $L$<li>Routing-Entscheidung (Routing OH) in Schaltelemente: $L \times R$<br />​<br />Insgesamt: $L + L \times R$ |
| **Packet Switching (Store-and-forward)<br />**<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2011.26.57.png" alt="截屏2020-07-25 11.26.57" style="zoom:80%;" /> | $(L-1) \times N + 1$            | $N-1$                              | $L \times R$                                                 |
| **Packet Switching (cut-through)** <br /><img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2011.54.56.png" alt="截屏2020-07-25 11.54.56" style="zoom:80%;" /> | $L$                             | $N-1$                              | $L \times R$                                                 |

