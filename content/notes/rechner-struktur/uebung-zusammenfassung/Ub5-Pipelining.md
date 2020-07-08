---
# Basic info
title: "Ub5-Pipelining"
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
        weight: 8
---

**Befehlspipelining**

Zerlegung der Ausführung einer Maschinenoperation in Teilphasen, die dann von hintereinander geschaltenen Verarbeitungseinheiten taktsynchron bearbeitet werden, wobei *jede Einheit genau eine spezielle Teiloperation ausführt.*

**Pipeline-Stufe**

Stufen, der Pipeline, die jeweils durch Pipeline-Register getrennt sind

**Pipelining**

- Pipeline-Stufen benutzen unterschiedliche Ressourcen
- Ausführung eines Befehls in $k$ Taktzyklen

- erhöht den Durchsatz, reduziert NICHT Ausführungszeit eines Befehls
- **Taktzyklus ist abhängig von der *langsamsten* Stufe**

- Unterscheidung zwischen Integer- und FP-Ausführung



## Bsp

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-02%2013.13.37.png" alt="截屏2020-07-02 13.13.37" style="zoom:80%;" />

- Zykluszeit

  - OHNE Pipelining

    $$
    \text{Zykluszeit} = \text{Summe aller Stufen}
    $$
    
    $$
    \text{Zykluszeit} = 250 + 100 + 130 + 220 + 50 = 750 \text{ps}
    $$
    
  - Mit Pipelining
    $$
    \text{Zykluszeit} = \text{Längste Stufe} + \text{Latenz des Pipelineregisters}
    $$
  
    $$
    \text{Zykluszeit} = 250 + 20 = 270 \text{ps}
    $$
  
    

- SpeedUp
  $$
  \begin{aligned} \text {SpeedUp} &=\frac{\text { average exec time WITHOUT pipeline }}{\text { average exectime WITH pipeline }} \\ &=\frac{\mathrm{CPI} \cdot \text { CycleTime WITHOUT pipeline}}{\mathrm{CPI} \cdot \text { CycleTime WITH pipeline}} \end{aligned}
  $$
  