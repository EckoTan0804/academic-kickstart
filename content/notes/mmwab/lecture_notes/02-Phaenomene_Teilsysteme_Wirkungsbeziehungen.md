---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 102

# Basic metadata
title: "Phänomene, Teilsysteme, Wirkungsbeziehungen"
date: 2021-03-17
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
        weight: 2

---

## Aufgabenteilung zwischen Mensch und Maschine

Technische Anlagen und Dienstleistungen werden heute überwiegend **rechnergestützt** betrieben.  Das heißt mit einer **Aufgabenteilung** zwischen dem Menschen als **Nutzer** bzw. **Bediener** und der technischen Anlage, kurz Maschine.

Die technische Anlage führt also selbsttätig Funktionen aus, die 

- von einem Menschen **angestoßen**,

- in ihrem Verlauf **überwacht** und

- fallweise **korrigiert** werden.

Die besondere Rolle der Aufgabenteilung ergibt sich aus dem Zweck des Einsatzes von Rechnern, dass diese komplexe **Teilaufgaben automatisch bearbeiten**.

**10 Ebenen der überwachten Automatisierung**

1. Bietet keine Unterstützung an: Der Mensch muss alles selbst machen.
2. Bietet eine vollständige Menge von Alternativen an und
3. schränkt diese auf wenige ein oder
4. schlägt die geeignetste davon vor oder
5. führt diesen Vorschlag aus, wenn der Mensch zustimmt oder
6. gesteht dem Menschen zu, bis zu einem bestimmten Zeitpunkt vor der automatischen Ausführung ein Veto einzulegen oder
7. führt die vorgeschlagene Handlung aus und benachrichtigt den Menschen darüber oder
8. benachrichtigt ihn nur, wenn er es wünscht oder
9. benachrichtigt ihn nach der Ausführung, wenn der Nachgeordnete sich dafür entscheidet.
10. Entscheidet immer und handelt autonom, ignoriert den Menschen als Kontrollinstanz.

### Die Rollen von Nutzer, Benutzer und Bediener/Betätiger

- **Nutzer** einer Maschine (Customer)

  diejenige Person, welche aus der Leistung der Maschine Nutzen zieht, ohne diese selbst betätigen zu müssen (Fahrgast in einem Taxi, Eigner einer Fabrik mit Fertigungsmaschinen)

- **Bediener/Betätiger** einer Maschine (Operator)

  diejenige Person, die eine Maschine in Gang setzt und so in Gang hält, dass sie für einen anderen Nutzen bringt (Taxifahrer, Maschinenbediener in einer Fabrik)

- **Benutzer** einer Maschine (User)

  diejenige Person, die eine Maschine in Gang setzt und so in Gang hält, dass sie selbst von deren Leistung Nutzen ziehen kann (privater Autofahrer, privater PC-Benutzer)

Taxi Bsp.:

![截屏2021-03-17 23.53.11](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-17%2023.53.11.png)

### Die Mensch als Nutzer-Maschine-Schnittstelle

Taxi Bsp.:

### ![截屏2021-03-18 00.00.41](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2000.00.41.png)

Die ideale Nutzer-Maschine-Schnittstelle hätte

- ein **quasi-menschliches** Gesicht gegenüber dem **Nutzer**
  - für die Äußerungen von Menschen empfänglich sein
  - die menschlichen Sinne erreichen
- ein **technisches** Gesicht gegenüber der **Maschine**.

### Die Sinne des Menschen

![截屏2021-03-18 00.04.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2000.04.23.png)

![截屏2021-03-18 00.04.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-18%2000.05.01-20210318000634754.png)

## Menschlichen Äußerung

Zweck der menschlichen Äußerung

- **Energieübertragung**
  - die **selbst aufgewendete Energie vollständig auf das Gegenüber zu übertragen**, um es in einen bestimmten Zielzustand zu bringen (Schieben, Ziehen, Biegen, Brechen, ...).
  - hauptsächlich aus der **Wechselwirkung mit leblosen Gegenständen** vertraut.
- **Nachrichtenübertragung**
  - **das Gegenüber zu veranlassen**, in einen bestimmten Zielzustand **mit eigener Energieaufwendung** zu gelangen.
  - hauptsächlich aus der **Wechselwirkung mit Lebewesen**, insbesondere mit anderen Menschen vertraut.

### Information

- **Syntaktische** Information
  - ist ein Maß für die kürzeste Kodierung einer Nachricht
  - braucht keinen Empfänger,
  - steckt objektiv in der Nachricht,
  - ist bestimmt durch ein Alphabet und die Auftrittswahrscheinlichkeit seiner Zeichen

- **Semantische** Information
  - bezeichnet die Bedeutung einer Nachricht für den Empfänger
  - ist nicht quantifizierbar
  - braucht einen Empfänger und entsteht erst bei diesem
  - ist bestimmt durch die Bedeutung für den Empfänger.

Schwerpunkte für die Mensch-Maschine-Wechselwirkung in der Anthropomatik

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2000.04.28.png" alt="截屏2021-03-20 00.04.28" style="zoom:80%;" />

### Kommando vs. Auftrag

|             | Kommando                                                     | Auftrag                                                      |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Vorgabe     | Zielerreichungsweges                                         | Ziel und Randbedingungen                                     |
| Überwachung | Der Kommandogeber überwacht das Einhalten der vorgegebene Schritte, die nach Kenntnis des Kommandogebers zum gewünschten Ziel führen | Der Auftraggeber überwacht die Einhaltung der Randbedingungen während, und den Grad der Zielerreichung nach der Auftragsbearbeitung |
| Empfänger   | "dummer" Kommandoempfänger                                   | "inteeligenter" Auftragnehmer                                |
| Wer gibt?   | Bediener / Betätiger                                         | Nutzer                                                       |

#### Vom Auftrag zur Aufgabe

- Der Nutzer als **Dienstnehmer** erteilt der Maschine als **Dienstgeber** (-leister) einen Auftrag.

- Der Dienstgeber nimmt den Auftrag als **Aufgabe** an und erfüllt den Auftrag durch

  Lösen dieser Aufgabe.

- Die Aufgabe gliedert sich in **Teilaufgaben**, die in einer bestimmten Reihenfolge abgearbeitet werden.

  - Teilaufgaben können als Unteraufträge Unter-Dienstgebern erteilt werden.

- Die Aufgabe wird untergliedert in Tätigkeiten, Handlungen usw. bis hinab zur Muskelaktion als dem letzlichen Ausdruck der Aufgabe

#### Von der Aufgabe zur Muskelaktion

![截屏2021-03-20 00.13.34](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2000.13.34.png)

### Vom Wahrnehmen zum Handeln

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2000.15.20.png" alt="截屏2021-03-20 00.15.20" style="zoom:80%;" />

- **Wahrnehmen**: Prozess der Aufnahme von äußeren Signalen.
- **Entscheiden**: die Wahl zwischen Handlungsalternativen aufgrund von Wahrnehmungsergebnissen.

- **Handeln**: das Aussenden von Signalen an die Umgebung.

#### Drei Ebenen des Verhaltens

![截屏2021-03-20 00.16.57](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2000.16.57.png)

![截屏2021-03-21 13.09.39](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2013.09.39.png)

> 💡 **Ausreichendes Training macht den Menschen sicherer und schneller**
>
> ![截屏2020-10-06 13.05.00](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-10-06%2013.05.00-20210321130832072.png)

### Das Wahrnehmen

Die Trias der Perzeption

![Stufen der Wahrnehmung: Logisch](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2000.30.54.png)

Logische Stufen der Wahrnehmung

![截屏2021-03-20 00.32.28](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2000.32.28.png)

Example

![截屏2021-03-20 00.32.51](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2000.32.51.png)

