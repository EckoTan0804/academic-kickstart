---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 107

# Basic metadata
title: "Qualitative Gestaltungsregeln, Normen, Richtlinien"
date: 2021-03-20
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Vorlesung", "Zusammenfassung", "MMWAB"]
categories: ["Lecture"]
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
    mmwab:
        parent: mmwab-lecture-note
        weight: 7

---

## DIN EN ISO 9241: Ergonomie der Mensch-System-Interaktion

Leitkriterien

- **Effektivität**

  Genauigkeit und Vollständigkeit, mit der die Benutzer ein bestimmtes Ziel erreichen.

- **Effizienz**

  Genauigkeit und Vollständigkeit des erzielten Effekts im Verhältnis zum Aufwand der Benutzer

- **Arbeitszufriedenheit (Akzeptanz)**

  Positive Einstellung der Benutzer gegenüber der Nutzung des Systems sowie ihre Freiheit von Beeinträchtigungen durch das System

🎯 Ziel: **Gebrauchstauglichkeit** 

(Maß der Effektivität, Effizienz und Arbeitszufrieden- heit, mit der ein Benutzer mit einem gegebenen System vorgegebene Ziele erreichen kann)

### Teil 110: Grundlagen der Dialoggestaltung

> See also: [Notes of DIN EN IS ISO 9241]({{< relref "../../gestaltungsgrundsaetze-fuer-interaktive-echtzeitsysteme/vorlesung/vl-05.md" >}})

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2010.16.42.png" alt="截屏2021-03-21 10.16.42" style="zoom:80%;" />

Ein gut gestalteter Dialog sollte folgende Eigenschaften haben

- **Aufgabenangemessenheit**

  - Unterstützt den Benutzer bei der Aufgaben-Erledigung
  - Anliegen
    - Minimieren der Interaktionsschritte, die für ein bestimmtes Ergebnis erforderlich sind. 
    - Nur die Information anzeigen, die im Kontext wichtig ist (Nutzinformation)

  - Bsp: 

    ![截屏2020-10-09 11.24.53](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-10-09%2011.24.53-20210321100909193.png)

- **Selbstbeschreibungsfähigkeit**

  - Jeder einzelne Dialogschritt ist durch Beschreibung oder Rückmeldung unmittelbar verständlich oder er wird auf Anfrage erklärt.

    > Dialog ist selbstbeschreibungsfähig, wenn der Benutzer zu jedem Zeitpunkt weiß:
    >
    > - an welcher Stelle im Dialog er sich befindet
    > - welche Handlungen unternommen werden können
    > - wie diese ausgeführt werden können

  - Bsp

    ![截屏2020-10-09 11.26.26](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-10-09%2011.26.26-20210321101005459.png)

- **Steuerbarkeit**

  - Der Benutzer soll in der Lage sein, den Dialogablauf zu steuern, d. h. Ablauf, Richtung und Geschwindigkeit zu beeinflussen, bis er sein Ziel erreicht hat.

- **Erwartungskonformität**

  - Der Dialog entspricht den Kenntnissen des Benutzers aus seinem Arbeits- gebiet, seiner Ausbildung und seiner Erfahrung. Außerdem ist der Dialog konsistent.
  - Bsp: Farben signalisieren Zustand (Rot und blinken --> Warnung)

- **Fehlertoleranz**

  - Trotz erkennbar fehlerhafter Eingaben kann das beabsichtigte Arbeitsergebnis mit keinem oder minimalem Korrektur- aufwand erreicht werden.

  - Bsp

    ![截屏2020-10-09 11.33.00](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-10-09%2011.33.00-20210321101246124.png)

- **Individualisierbarkeit**

  - Der Benutzer kann den Dialog an seine Arbeitsaufgabe sowie seine individuellen Fähigkeiten und Vorlieben anpassen.

  - Bsp

    ![截屏2020-10-09 11.36.14](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-10-09%2011.36.14-20210321101436322.png)

- **Lernförderlichkeit**

  - Der Benutzer wird beim Erlernen der Anwendung unterstützt und angeleitet.

  - Bsp

    ![截屏2020-10-09 11.30.57](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-10-09%2011.30.57-20210321101602984-20210321102615521.png)







## 8 goldenen Regeln von Ben Shneiderman

Qualitative Regeln für die Gestaltung der Mensch-Computer-Schnittstelle

1. Strebe nach Konsistenz (**Strive for consistency**)

   > Similar sequences of actions should have similar terminology in prompts, actions, menus, help and commands.

2. Biete häufigen Benutzern Abkürzungen (»shortcuts«) an (**Enable frequent users to use shortcuts**.)

   > As the frequency of use increases, so do the user's desires to reduce the number of interactions and to increase the pace of interaction. Abbreviations, function keys, hidden commands, and macro facilities are very helpful to an expert user.

3. Biete informative Rückmeldungen an (**Offer informative feedback**)

   > For every operator action, there should be some system feedback.

4. Entwerfe abgeschlossene Dialoge (**Design dialog to yield closure**)

   > Sequences of actions should be organized into groups with a beginning, middle, and end. The informative feedback at the completion of a group of actions gives the operators the satisfaction of accomplishment, a sense of relief, the signal to drop contingency plans and options from their minds, and an indication that the way is clear to prepare for the next group of actions.

5. Biete einfache Fehlerbehandlung (**Offer simple error handling**)

   > As much as possible, design the system so the user cannot make a serious error. If an error is made, the system should be able to detect the error and offer simple, comprehensible mechanisms for handling the error.

6. Erlaube einfache Rücksetzmöglichkeiten (**Permit easy reversal of actions**)

   > This feature relieves anxiety, since the user knows that errors can be undone; it thus encourages exploration of unfamiliar options. The units of reversibility may be a single action, a data entry, or a complete group of actions.

7. Lasse den Benutzer Aktionen initiieren und ihn nicht nur reagieren (**Support internal locus of control**)

   > This refers to giving users the sense that they are in full control of events occurring in the digital space. Experienced operators strongly desire the sense that they are in charge of the system and that the system responds to their actions. Design the system to make users the initiators of actions rather than the responders.

8. Halte die Belastung des Kurzzeitgedächtnisses gering (**Reduce short-term memory load**)

   > The limitation of human information processing in short-term memory requires that displays be kept simple, multiple page displays be consolidated, window-motion frequency be reduced, and sufficient training time be allotted for codes, mnemonics, and sequences of actions.

Summary:

![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/8-golden-rules-of-user-interface-design.png)



Resource

- [Shneiderman's "Eight Golden Rules of Interface Design"](https://faculty.washington.edu/jtenenbg/courses/360/f04/sessions/schneidermanGoldenRules.html)
- [Ben Shneiderman: The ‚Eight Golden Rules‘ of Interface Design](https://xd-i.com/user-interface-design/ui-ux-design-course/ben-shneiderman-the-eight-golden-rules-of-interface-design/)
- [Shneiderman’s Eight Golden Rules Will Help You Design Better Interfaces](https://www.interaction-design.org/literature/article/shneiderman-s-eight-golden-rules-will-help-you-design-better-interfaces)

## 7 Grundregeln von Max Syrbe

1. **Beachte die Eigenschaften der Sinnesorgane**

   Z.B. Gesichtsfeld, Sehschärfe, Hörfläche, Zeitauflösung

2. **Wähle die Prozesszustandsdarstellung aufgabenabhängig**

   Z.B. für genaue Ablesung digital, für Tendenzablesung analog, für Ablesung von Grenzüberschreitungen binärer Wechsel von Farbe, Symbol/Piktogramm oder Frequenz

3. **Wähle eine der Aufgabe direkt entsprechende Darstellung**

   Z.B. Prozessbild statt »Uhrenladen«, Drehrichtung statt »+, -«-Tasten

4. **Vermeide hinsichtlich der Aufgabenstellung unnütze Information (Störinformation)**

5. **Beachte die unbewusste Aufmerksamkeitssteuerung des Menschen**

6. **Beachte populationsstereotype Erwartungen**

   Z.B. Potentiometer nach rechts gibt größere Werte

7. **Gestalte zusammengehörige Anzeige- und Bedienelemente auffällig gleich und nicht zusammengehörige besonders ungleich**

## Beispiele