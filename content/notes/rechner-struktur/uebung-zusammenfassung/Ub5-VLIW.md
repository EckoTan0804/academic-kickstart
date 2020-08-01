---
# Basic info
title: "Ub5-VLIW Prozessoren"
linktitle: "Ub5-VLIW"
date: 2020-07-20
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
        weight: 11

weight: 153
---

## Überblick

- Befehlswort aus **mehreren einzelnen Befehlen** zusammengesetzt

- Parallelität **explizit vom Compiler** angegeben 
- **Statisches** Konzept

- ”Platzhalter“ in Befehlswort für jede vorhandene Ausführungseinheit

- Sinnvoll bei Spezialanwendungen: DSP, Graphik, Netzwerk 

- Moderne Varianten: **EPIC** (Intel Itanium), Transmeta Crusoe

- Befehlformat

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-20%2014.28.15.png" alt="截屏2020-07-20 14.28.15" style="zoom:50%;" />

## Aufgabe

Der folgende Assembler-Code soll auf einem VLIW-Prozessor mit drei parallelen Ausfüh- rungseinheiten ausgeführt werden. Geben Sie hierfür eine möglichst effiziente Befehlsvertei- lung an. Die Befehle können beliebig umsortiert werden, so lange die Korrektheit der Anwen- dung gewährleistet ist.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-20%2014.31.19.png" alt="截屏2020-07-20 14.31.19" style="zoom:50%;" />

> Abhängigkeit: 
>
> - Befehl 3 is abhängig von Befehl 2
>
> - Befehl 4 und 5 sind abhängig von Befehl 3
> - Befehl 8 und 9, 10 sind abhängig von Befehl 7
> - Befehl 9 is abhängig von Befehl 6

### Teilaufgabe (a)

Nehmen Sie an, dass der Prozessor über drei Ausführungseinheiten verfügt, die jeweils alle Befehle ausführen können.

#### Zuordnung Befehl 3, 4, 5: Abhängigkeiten beachten

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-20%2015.20.44.png" alt="截屏2020-07-20 15.20.44" style="zoom:80%;" />

{{% alert note %}} 

Abhängigkeit

$\Rightarrow$ Befehl muss in **nächste** VLIW (also nächste Zeile)

{{% /alert %}}

#### Zuordnung Befehl 6, 7, 8

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-20%2015.19.52.png" alt="截屏2020-07-20 15.19.52" style="zoom: 80%;" />

{{% alert warning %}} 

Zuordnung Befehl 8

- Befehle 9 und 10 auch von Befehl 7 abhängig
- führt zu **langer** Befehlsfolge und einem nötigen 5. Befehl 🤪

$\Rightarrow$ Optimierung notwendig! 💪

{{% /alert %}}

Da Befehl 6 (`ld r9, [r7]`) und Befehl 7 (`ld r11, [r12]`) nicht abhängig sind, könne die beide Befehlen vertauscht werden. Und wir haben die Neuordnung von Befehl 8.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-20%2015.27.41.png" alt="截屏2020-07-20 15.27.41" style="zoom:80%;" />

#### Zuordnung restlicher Befehle

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-20%2015.28.24.png" alt="截屏2020-07-20 15.28.24" style="zoom:80%;" />