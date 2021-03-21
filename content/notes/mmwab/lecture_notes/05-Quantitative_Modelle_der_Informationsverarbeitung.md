---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 105

# Basic metadata
title: "Quantitative Modelle der Informationsverarbeitung"
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
        weight: 5

---

![截屏2021-03-20 14.31.21](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2014.31.21.png)

## Der "Model Human Processor" (MHP) nach Card, Moran & Newell (1983)

![截屏2021-03-21 15.39.01](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-21%2015.39.01.png)

>  ![截屏2020-10-15 10.40.53](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-10-15%2010.40.53-20210320214746248.png)

|                               | Zykluszeit $\tau$              | Einheit                               | Kapazität $\mu$     | Halbwertszeit $\delta$ | Kodierungsform  $\kappa$ |
| ----------------------------- | ------------------------------ | ------------------------------------- | ------------------- | ---------------------- | ------------------------ |
| **PerzeptiverProzessor**      | $\tau\_P = $ 100 [50 - 200] ms |                                       |                     |                        |                          |
| **Visuelle Sinnesgedächtnis** |                                | Zeichen                               | 17 [7 - 17] Zeichen | 200 [70 - 1000] ms     | physikalisch             |
| **Arbeitsgedächtnis**         |                                | Merkeinheit (ME)                      | 7 [5 - 9] ME        | 7 [5 - 226] s          | visuell / akustisch      |
| **Langzeitgedächtnis**        |                                | aktivierte logische Erinnerungsstücke | $\infty$            | $\infty$               | semantisch               |
| **Kognitiver Prozessor**      | $\tau\_C = $ 70 [25 - 170] ms  |                                       |                     |                        |                          |
| **Motor Prozessor**           | $\tau\_M = $70 [30 - 170] ms   |                                       |                     |                        |                          |



- **Perzeptiver Prozessor (Sensor Prozessor)**

  - sorgt dafür, dass Reize, die uns über die Sinne erreichen, in einen sinnesnahen Speicher, das sogenannte **Sinnesgedächtnis** geschrieben werden.
  - Zykluszeit: 100 [50 - 200] ms

- **Visueller Speicher **("Register")

  - Visuelle Sinnesgedächtnis

    - Einheit: Zeichen

    - Kapazität: 17 [7 - 17] Zeichen

    - Halbwertszeit: 200 [70 - 1000] ms

      (Zeit, in der die Wahrscheinlichkeit, dass der Speicherinhalt erinnert werden kann kleiner als 0,5 wird)

    - Kodierungsform: physikalisch (die Bildfigur ist als elektrisches Erregungsmuster repräsentiert.)

- **Arbeitsgedächtnis** ("Arbeitsspeicher")

  - Einheit: **Merkeinheit (ME)**, also aktivierte logische Erinnerungsstücke im Langzeitgedächtnis

    - "Chunk" als Merkeinheit im Arbeitsgedächtnis

      ![截屏2021-03-20 15.10.37](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2015.10.37.png)

  - Kapazität: 7 [5 - 9] ME

  - Halbwertszeit: 7 [5 - 226] s

  - Kodierungsform: visuell / akustisch

- **Langzeitgedächtnis** ("Festplatte")

  - Einheit: aktivierte logische Erinnerungsstücke 
  - Annahme: "unendliche" Kapazität ("unendlich" = bisher keine obere Schranke für die Kapazität feststellbar)
  - Halbwertszeit: $\infty$
  - Kodierung: semantisch

- **Kognitiver Prozessor**
  - Zykluszeit: 70 [25 - 170] ms

- **Motorische Prozessor**
  - Zykluszeit: 70 [30 - 170] ms

### "Recognize-Act Cycle" im Arbeitsgedächtnis

![截屏2021-03-20 15.18.59](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2015.18.59.png)

Example

![截屏2021-03-20 15.20.14](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2015.20.14.png)

![截屏2021-03-20 15.20.37](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2015.20.37.png)

![截屏2021-03-20 15.20.58](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2015.20.58.png)

![截屏2021-03-20 15.21.19](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2015.21.19.png)



## 10 Operationsprinzipien des MHP

0. **"Recognize-Act" Cycle des kognitiven Prozessors**

   Mit jedem Zyklus des kognitiven Prozessors initiiert der Inhalt des Arbeitsgedächtnisses mit ihm assoziierte Aktionen im Langzeitgedächtnis; diese Aktionen wiederum verändern den Inhalt des Arbeitsgedächtnisses

1. **Prinzip der variablen Taktrate des perzeptiven Prozessors**

   Der perzeptive Prozessor arbeitet umso schneller, je intensiver der Stimulus ist.

2. **Prinzip der kontextabhängigen Kodierung (Encoding Specifity Principle)**

   Der Kontext der Kodierung im Langzeitgedächtnis bestimmt,

   was gespeichert wird und das bestimmt wiederum, wie etwas wieder- gefunden wird. *Beispiel: Der Begriff »Bank« kann assoziiert mit »Geld« oder »Sitzen« gespeichert bzw. wiedergefunden werden.*

3. **Prinzip der Unterscheidung (Discrimination Principle)**

   Die Schwierigkeit des Wiederauffindens einer Merkeinheit im Langzeit- gedächtnis ist bestimmt durch die Kandidaten im Langzeitgedächtnis, die mit dieser Merkeinheit assoziiert sind. (Oder: Je mehr Kandidaten mit einer bestimmten Merkeinheit assoziiert sind, desto größer ist die Gefahr, dass beim Abrufen Verwechslung eintritt.)

4. **Prinzip der variablen Taktrate des kognitiven Prozessors** 

   Die Taktrate des kognitiven Prozessors ist umso höher, je mehr Anstrengung durch gesteigerte Anforderungen aus der Aufgabe aufgewendet werden muss; sie steigt auch mit wachsender Übung.

5. **Fitts's Law**

   Die Zeit, die benötigt wird, um die Hand auf ein Zielfeld der Weite $W$ zu bewegen, das in einem Abstand $D$ von den Hand entfernt liegt, beträgt
   $$
   T\_{pos} = I\_M \log\_2(\frac{D}{W} +  0.5)
   $$

   - $I\_M = $ 100 [70 - 120] ms

   > Ergänzung: Steering Law (Gehört nicht zu den 10 Operationsprinzipien)
   >
   > ![截屏2021-03-20 21.50.59](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2021.50.59.png)

6. **Das Potenzgesetz der Übung (Power Law of Practice)**

   Die Zeit $T\_{n}$, die für die $n$-te Wiederholung einer Aufgabe benötigt wird, folgt dem Potenzgesetz
   $$
   T\_n = T\_1 n^{-\alpha}
   $$

   - $\alpha$ = 0.4 [0.2 - 0.6]

   ![截屏2021-03-20 21.55.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2021.55.48.png)

7. **Prinzip der Wahlunsicherheit (Uncertainty Principle)**

   Die Zeit $T\_{Wahl}$, für die Entscheidung zwischen $n$ Alternativen, hängt von der Unsicherheit über diese Alternativen ab, ausgedrückt als Entropie $H$:
   $$
   T\_{wahl} = C + kH \quad (\text{Hick's Law})
   $$

   - $C$: Konstante für $\tau\_P + \tau\_M$
   - $k = $ 150 [0 -  150] ms (experimentell ermittelt; wird mit Übung kleiner)
   - $\mathrm{H}=\sum\_{i=1}^{n} p\_{i} \log \_{2}\left(1+\frac{1}{p\_{i}}\right)$
     - $p\_i$: Eintrittswahrscheinlichkeit der Alternative $i$

   Bsp

   ![截屏2021-03-20 22.50.34](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2022.50.34.png)

8. **Prinzip des verständigen Verhaltens (Rationality Principle)**

   Ein Mensch handelt so, dass er seine Ziele durch verständiges Verhalten erreicht, gegeben die Struktur der Aufgabe sowie deren Informations- eingang und begrenzt durch sein Wissen und seine Verarbeitungsfähigkeit.

   ![截屏2021-03-20 22.52.16](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-03-20%2022.52.16.png)

   (Siehe [GOMS-Modell](#das-goms-modell))

9. **Prinzip des Problemraums (Problem Space Principle)**

   Verständige Tätigkeit von Menschen zur Lösung von Problemen kann beschrieben werden durch:

   - Ein Menge von Wissenszuständen
   - Operationen, um einen Wissenszustand in einen anderen zu überführen
   - Bedingungen zur Anwendung von Operationen
   - Steuerwissen für die Entscheidung, welche Operation als nächste kommt

## Das GOMS-Modell

- **G**oals
  - Ziele, die mit einer Operation verfolgt werden (*Was zu tun ist?*)
  - z.B. "Ein Zeichen löschen"
- **O**perators
  - Operatoren, die für die Zielerreichung benutzt werden (*Wie es getan werden kann*)
  - z. B. ein Tastaturbefehl oder eine Menüauswahl)
- **M**ethods: Die Methoden, eine Folge von Operatoren einzusetzen, um ein bestimmtes Ziel zu erreichen.
- **S**election Rules
  - Die Regeln, nach denen Operatoren oder Methoden ausgewählt werden
  - z. B. für Kopieren: "Ctrl-C" oder "Hauptmenü > Bearbeiten > Kopieren" oder "Rechte Maustaste > Kopieren"