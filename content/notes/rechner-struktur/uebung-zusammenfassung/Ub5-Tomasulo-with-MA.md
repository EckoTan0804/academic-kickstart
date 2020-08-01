---
# Basic info
title: "Ub5-Tomasulo Algorithm with Memory Access (MA)"
linktitle: "Ub5-Tomasulo Algorithm (MA)"
date: 2020-07-15
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
        weight: 10

weight: 152
---

## Annahme

- Pipeline: 
  - Fetch (IF)
  - Decode (ID)
  - Issue + Renaming (IS)
  - Execute oder Memory Access (EX/MA)
  - Write Back (WB)

- Einheiten:

  - 2 Lade-Speichereinheiten (Load/Store Unit)

  - 1 Integer-Additionseinheit

  - 1 Integer-Multiplikationseinheit

  - 1 FP-Additionseinheit

    | Einheit                       | L/S  | Int-Add/Sub | Int-Mul | FP-Add |
    | ----------------------------- | ---- | ----------- | ------- | ------ |
    | Anzahl                        | 2    | 1           | 1       | 1      |
    | Bearbeitungsdauer (in Takten) | 3    | 1           | 3       | 2      |

- Statische Sprungvorhersage mit fortwährendem Füllen der Pipeline vom Sprungziel; dafür sei ein Sprungzieladresscache vorhanden und die Sprungvorhersage laute auf Taken

- FP-Register und normale Register können gleichzeitig in der WB-Stufe beschrieben werden

- Schreiben in Speicher geschehe nicht üßber CDB 

- Volles Bypassing

- Die Befehlszuordnungs- und Rückordnungsbandbreite betrage **4** Befehle; **zwei** Befehle werden pro Takt maximal geholt
- Entsprechend gibt es **zwei Dekodiereinheiten**, die gleichzeitig arbeiten können
- Die Auswertung der Sprungzieladresse erfolge in der Stufe Execute der Int-Add-Einheit; das Schreiben des Befehlsza ̈hlers (Instruction Counter, Program Counter) in der WB-Stufe
- Speicherlesezugriffe erfolgen analog zu normalen Rechenoperationen in der Ausführungsstufe, das Rückschreiben geschieht dabei als separater Schritt (WB)
- Speicherschreibzugriffe haben eine Ausführungsdauer von 3 Takten



## Aufgabe

Folgender Code werde darauf ausgeführt, wobei 

- `R0 = 0`
- `R1` eine Speicheradresse, 
- `R2 = R1 + 24` 
- `F2` sei beliebig

```assembly
LOOP: LD.D F0,0(R1)   ; loads Mem[i]
      ADD.D F4,F0,F2  ; adds to Mem[i]
      S.D 0(R1),F4    ; stores into Mem[i]
      ADD R1,R1,#8    ;
      SUB R3,R1,R2    ; R3 = R1-R2
      BLTZ R3,LOOP    ; branch if R1 < R2
```

## Verlauf der Pipeline

| Befehl           | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | 17   | 18   | 19   | 20   | 21   | 22   |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| `LD.D F0,0(R1)`  | IF   | ID   | IS   | M    | M    | M    | WB   |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| `ADD.D F4,F0,F2` | IF   | ID   | IS   |      |      |      | EX   | EX   | WB   |      |      |      |      |      |      |      |      |      |      |      |      |      |
| `S.D 0(R1),F4`   |      | IF   | ID   | IS   |      |      |      |      | M    | M    | M    |      |      |      |      |      |      |      |      |      |      |      |
| `ADD R1,R1,#8`   |      | IF   | ID   | IS   | EX   | WB   |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| `SUB R3,R1,R2`   |      |      | IF   | ID   |      | IS   | EX   | WB   |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| `BLTZ R3,LOOP`   |      |      | IF   | ID   |      |      |      | IS   | EX   | WB   |      |      |      |      |      |      |      |      |      |      |      |      |
| `LD.D F0,0(R1)`  |      |      |      | IF   | ID   |      | IS   | M    | M    | M    | WB   |      |      |      |      |      |      |      |      |      |      |      |
| `ADD.D F4,F0,F2` |      |      |      | IF   | ID   |      |      | IS   |      |      | EX   | EX   | WB   |      |      |      |      |      |      |      |      |      |
| `S.D 0(R1),F4`   |      |      |      |      | IF   | ID   |      |      |      |      | IS   |      | M    | M    | M    |      |      |      |      |      |      |      |
| `ADD R1,R1,#8`   |      |      |      |      | IF   | ID   |      |      |      | IS   | EX   | WB   |      |      |      |      |      |      |      |      |      |      |
| `SUB R3,R1,R2`   |      |      |      |      |      | IF   | ID   |      |      |      |      | IS   | EX   | WB   |      |      |      |      |      |      |      |      |
| `BLTZ R3,LOOP`   |      |      |      |      |      | IF   | ID   |      |      |      |      |      |      | IS   | EX   | WB   |      |      |      |      |      |      |
| `LD.D F0,0(R1)`  |      |      |      |      |      |      | IF   | ID   |      |      | IS   |      | M    | M    | M    | WB   |      |      |      |      |      |      |
| `ADD.D F4,F0,F2` |      |      |      |      |      |      | IF   | ID   |      |      |      |      |      |      |      | EX   | EX   | WB   |      |      |      |      |
| `S.D 0(R1),F4`   |      |      |      |      |      |      |      | IF   | ID   |      |      |      |      |      | IS   |      |      | M    | M    | M    |      |      |
| `ADD R1,R1,#8`   |      |      |      |      |      |      |      | IF   | ID   |      |      |      |      |      |      | IS   | EX   | WB   |      |      |      |      |
| `SUB R3,R1,R2`   |      |      |      |      |      |      |      |      | IF   | ID   |      |      |      |      |      |      |      | IS   | EX   | WB   |      |      |
| `BLTZ R3,LOOP`   |      |      |      |      |      |      |      |      | IF   | ID   |      |      |      |      |      |      |      |      |      | IS   | EX   | WB   |

### Takt 1

`LD.D` und `ADD.D` geholt

### Takt 2

- `LD.D` und `ADD.D` dekodiert

- `SD.D` und `ADD` geholt

- RS noch leer

- Rückordnungspuffer

  | Befehlsnr. | Ziel | Quelle |
  | ---------- | ---- | ------ |
  |            |      |        |

- Registerstatustabelle

  | Field | R1   | R2   | R3   | F0   | F2   | F4   |
  | ----- | ---- | ---- | ---- | ---- | ---- | ---- |
  | Value | (R1) | (R2) | (R3) | -    | (F2) | (F4) |
  | RS    |      |      |      |      |      |      |

- Befehlsfenster

  | Nummer | Befehl           | Stage |
  | ------ | ---------------- | ----- |
  | 1      | `LD.D F0,0(R1)`  | ID    |
  | 2      | `ADD.D F4,F0,F2` | ID    |

### Takt 3 & 4

- `SUB` und `BLTZ` geholt (Takt 3), dekodiert (Takt 4)

- `LD.D` und `ADD.D` geholt (Takt 4)

- Befehlsfenster

  | Nummer | Befehl           | Stage |
  | ------ | ---------------- | ----- |
  | 1      | `LD.D F0,0(R1)`  | M     |
  | 2      | `ADD.D F4,F0,F2` | IS    |
  | 3      | `S.D 0(R1),F4`   | IS    |
  | 4      | `ADD R1,R1,#8`   | IS    |
  | 5      | `SUB R3,R1,R2`   | ID    |
  | 6      | `BLTZ R3,LOOP`   | ID    |

- Rückordnungspuffer

  | Befehlsnr. | Ziel | Quelle       |
  | ---------- | ---- | ------------ |
  | 4          | R1   | Int-Add      |
  | 3          | null | any L/S Unit |
  | 2          | F4   | FP-Add       |
  | 1          | F0   | L/S 1        |

- Reservation Stations:

  | Unit    | Empty | InFu | Op      | Dest | Src1       | Vld1 | RS1  | Src2 | Vld2 | RS2    |
  | ------- | ----- | ---- | ------- | ---- | ---------- | ---- | ---- | ---- | ---- | ------ |
  | L/S 1   | 0     | 1    | `ld.d`  | F0   | **[(R1)]** | 1    | 0    | -    | -    | -      |
  | L/S 2   | 0     | 0    | `s.d`   | (M)  | R1         | 1    | 0    | -    | 0    | FP-Add |
  | Int-A/S | 0     | 0    | `add`   | R1   | (R1)       | 1    | 0    | 8    | 1    | -      |
  | Int-Mul | 1     |      |         |      |            |      |      |      |      |        |
  | FP-Add  | 0     | 0    | `add.d` | F4   | -          | 0    | L/S1 | (F2) | 1    | -      |

  > Im moment gibt es keine Integer Multiplikation Operation. d.h., es gibt keine Eintrag  in RS `Int-Mul`, also die RS `Int-Mul` is leer (empty). Daher das `Empty` Feld von RS `Int-Mul` ist 1.
  >
  > Andere RS sind von Operationen belegt. Daher nicht leer $\Rightarrow$ `Empty=0`

- Registerstatustabelle

  | Field | R1      | R2   | R3   | F0        | F2   | F4     |
  | ----- | ------- | ---- | ---- | --------- | ---- | ------ |
  | Value | -       | (R2) | (R3) | -         | (F2) |        |
  | RS    | Int-A/S |      |      | **L/S 1** |      | FP-Add |

> Der erste Befehl `LD.D F0, 0(R1) ` lädt die Daten der Speicheradresse `R1` in Register `F0`. 
>
> In Takt 4 befindet `LD.D` in M (Memory Access, MA) Stufe. 
>
> Daher 
>
> - in RS `L/S 1`: 
>   - `inFu=1` (da schon in Execution Unit ausgeführt wird)
>   - `Src1 = [(R1)]` 
>   - Daten im `R1` ist gültig $\Rightarrow$ `Vld=1`
> - in Register `F0`: `RS=L/S 1` (Der Wert für Register `F0` wird von RS `L/S 1` berechnet/produziert)

### Takt 5 & 6

- `ADD` wird ausgeführt (Takt 5) und schreibt `R1` (Takt 6)
- Zuordnung von `SUB` noch im gleichen Takt (Takt 6)

- Befehlsfenster

  | Nummer | Befehl           | Stage |
  | ------ | ---------------- | ----- |
  | 1      | `LD.D F0,0(R1)`  | M     |
  | 2      | `ADD.D F4,F0,F2` | IS    |
  | 3      | `S.D 0(R1),F4`   | IS    |
  | 4      | `ADD R1,R1,#8`   | WB    |
  | 5      | `SUB R3,R1,R2`   | IS    |
  | 6      | `BLTZ R3,LOOP`   | ID    |
  | 7      | `LD.D F0,0(R1)`  | ID    |
  | 8      | `ADD.D F4,F0,F2` | ID    |
  | 9      | `S.D 0(R1),F4`   | ID    |
  | 10     | `ADD R1,R1,#8`   | ID    |

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-19%2022.46.16.png" alt="截屏2020-07-19 22.46.16" style="zoom:80%;" />

> Die Ausführung von dem 4. Befehl `ADD R1, R1, #8` ist fertig (in Rückordnungspuffer mit <span style="color:green">green</span> markiert). 
>
> Das Ergebnis `(R1) + 8` wird im Register `R1` geschrieben und das `RS` Feld von Register `R1` wird gelöscht. 
>
> Die RS `Int-A/S` ist wieder frei. Daher wird der 5.Befehl `SUB R3,R1,R2` eingetragen.

### Takt 7 & 8

- Erste `LD.D` beendet, Weiterleitung an `FP-ADD` (Takt 7)
- `ADD.D` Ausführung begonnen (Takt 7), **da wir volles Bypassing (Forwarding) haben**
- `SUB` angestoßen und berechnet (Takt 7)
- `BLTZ` zugeteilt (Takt 8)
- Zweites `LD.D` beginnt Ausführung (Takt 8)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-19%2022.50.45.png" alt="截屏2020-07-19 22.50.45" style="zoom:67%;" />

> Befehl 5 `SUB R3,R1,R2` ist fertig. ((in Rückordnungspuffer mit <span style="color:green">green</span> markiert))

{{% alert warning %}} 

Die vollendeten Add- und Subbefehle können noch nicht zurückgeordnet werden, da die vorherigen Befehle noch nicht alle beendet sind.

{{% /alert %}}

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-19%2022.52.23.png" alt="截屏2020-07-19 22.52.23" style="zoom:80%;" />

