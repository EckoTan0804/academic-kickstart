---
# Basic info
title: "Tumasulo Algorithm (Vorlesung)"
date: 2020-07-15
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
        parent: prozessor-techniken
        weight: 7

# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 26
---



## Stufen/Phasen von Superskalarer Pipelines

1. **Dekodierung**

   Erkennen der Befehlsklasse, der Adressierungsmodi usw.

2. **Registerumbenennung**

   Belegen eines physischen Registers mit dem virtuellen

3. **Befehlszuordnung**

   Eintragung in die Reservation Station 

   - Zuordnung eines Befehls zu einer Einheitenfamilie (**Issue**) und 
   - Anstoßen bei Verfügbarkeit der Operandenwerte (**Dispatch**)

4. **Ausführung**

   Ausführung der Rechenoperation oder des Speicherzugriffs

5. **Rückordnung**

   Schreiben des physischen Registers

{{% alert note %}} 

**Dekodierung** und **Registerumbenennung** können dabei zusammengefasst werden.

{{% /alert %}}



## **Algorithmus von Tomasulo**

- Um Datenabhängigkeit zu lösen

- Anstoßen der Befehle **dezentral** aus den Reservation Stations

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-15%2023.06.28.png" alt="截屏2020-07-15 23.06.28" style="zoom:80%;" />

  - `Empty`: Ist im moment dieser Platz frei (also keine RS Operation eingetragen)?
  - `InFU `: Befindet die Operation schon in Ausführungseinheit?
  - `Op`: Welche Operation handelt sich?
  - `Dest`: Zielregister
  - 1st Operand:
    - `Src1`: Wert des erstes Operands
    - `Vld1`: Ist der Wert im Moment gültig in Registertabelle?
    - `RS1`: Falls nicht gültig, durch welcher Eintrag in RS wird der benötige Wert berechnet?
  - 2nd Operand (`Src2`, `Vld2`, `RS2`): ähnlich wie erster Operand

- Einlesen der Operanden **dezentral** in die Reservation Stations

- Übertragung der Operandenwerte über den **Common Data Bus** (CDB): Tupel aus Tag/Wert

- Result Forwarding ebenfalls **dezentral**

- Einheiten können zu **Einheitenfamilien** zusammengefasst werden mit einer gemeinsamen, mehrzeiligen Reservation Station Table

- Reorder Buffer für die Befehlsrückordnung (Wiederherstellung der sequentiellen Programmsemantik)

  - Notwendig bei Sprungvorhersagen

### Bestandteile der Implementierung

#### Befehlsfenster

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-15%2023.18.17.png" alt="截屏2020-07-15 23.18.17" style="zoom:67%;" />

- Zuständig für die Bereitstellung *dekodierter* Befehle an Zuweisungseinheit (**Instruction Issue**)
- **Begrenzte** Größe
- Die hier gespeicherten Befehle sind **frei** von Namens-Abhängigkeiten und Steuerflussabhängigkeiten

- Hier erweitert um Feld der derzeitigen Stufe

#### Registerstatustabelle

- Gibt bei Verwendung von Register-Renaming auch Auskunft u ̈ber Abbildung von Architekturregistern auf physische Register

- Bsp:

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-15%2023.22.20.png" alt="截屏2020-07-15 23.22.20" style="zoom:80%;" />

  - Register R2
    - Im moment befindet sich kein Wert
    - Daher nicht gültig (`valid = 0`)
    - `RS=1`: Der neue Wert wird von RS1 berechnet

#### Reservation Stations (Reservierungstabelle)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-15%2023.26.06.png" alt="截屏2020-07-15 23.26.06" style="zoom:80%;" />

- Bei Tomasulo dezentral an jeder Einheit oder Familie von Einheiten
- Jede Ausführungseinheit kann über mehrere Einträge verfügen

#### Rückordnungspuffer

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-15%2023.27.22.png" alt="截屏2020-07-15 23.27.22" style="zoom:80%;" />

- Wird bei Eintragung eines Befehls in Reservation Station gefüllt 

- Enthält Zielregister, Befehlsnummer sowie produzierende Einheit

- Wird von **Retirement Unit** (Rückordnungsstufe) benutzt, um **Commitment** oder **Removement** durchzuführen

  ⇒ Rückordnung der Befehle in Programmreihenfolge