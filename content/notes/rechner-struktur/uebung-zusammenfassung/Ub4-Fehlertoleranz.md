---
# Basic info
title: "Ub4-Fehlertoleranz"
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
        weight: 6

weight: 140
---

## Blockdiagramm und Strukturformel

![截屏2020-06-25 09.57.52](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-25%2009.57.52.png)

- Systemfunktion (Strukturformel)
  $$
  S = B \wedge H \wedge R \wedge S \wedge (K_1 \vee K_2 \vee K_3)
  $$

- Fehlerbaum: Strukturbaum der **Negation** der Systemfunktion
  $$
  \begin{array}{ll}
  \neg S &=\neg\left(B \wedge H \wedge R \wedge S \wedge\left(K_{1} \vee K_{2} \vee K_{3}\right)\right) \\\\
  &=\neg B \vee \neg H \vee \neg R \vee \neg S \vee \neg\left(K_{1} \vee K_{2} \vee K_{3}\right) \\\\
  &=\neg B \vee \neg H \vee \neg R \vee \neg S \vee\left(\neg K_{1} \wedge \neg K_{2} \wedge \neg K_{3}\right)
  \end{array}
  $$
  ![截屏2020-06-25 10.00.55](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-25%2010.00.55.png)

### Funktionswahrscheinlichkeit

**1-aus-n Systeme** (Bsp: n =3)

- Das System fällt aus $\Leftrightarrow$ Alle 3 Komponenten $K_1$ bis $K_3$ fallen aus

  - Ausfallwahrscheinlichkeit einzelner Komponent $K$:
    $$
    \varphi(\neg K) = 1- \varphi(K)
    $$

  - $\Rightarrow$ Wahrscheinlichkeit, dass alle 3 Komponenten $K_1$ bis $K_3$ ausfallen, ist $(1- \varphi(K))^3$

- Funktionswahrscheinlichkeit des System:
  $$
  \varphi(S) = 1 - (1- \varphi(K))^3
  $$
  

> Funktionswahrscheinlichkeit des obigen Diagramms ist:
> $$
> \varphi=\underbrace{\varphi(B) * \varphi(B) * \varphi(R)}\_{\text {Seriensystem }} * \underbrace{\left(1-(1-\varphi(K))^{3}\right)}\_{\text {Parallelsystem }}
> $$

**Allgemein n-aus-m System**

- System funktionfähig $\Leftrightarrow$ Mindeste $n$ aus $m$ Komponente funktionfähig
- Bsp: 2-aus-3 System
  - System funktionsfähig, wenn die Komponenten 1&2, 1&3, 2&3 oder 1&2&3 funktionsfähig sind 
  - System NICHT funktionsfähig, wenn nur Komponente 1, 2, oder 3 funktionsfähig ist

- Funktionsfähigkeit
  $$
  \varphi_{m}^{n}=\sum_{k=n}^{m}\left(\begin{array}{c}
  m \\\\
  k
  \end{array}\right) * \varphi(K)^{k} *(1-\varphi(K))^{(m-k)}
  $$

  - $\varphi(K)$: Funktionswahrscheinlichkeit der Komponenten
  
  > Bsp: Aufg1 2(a)
  
- Systeme mit **Mehrheitsentscheider**
  $$
  \varphi_{m}^{n}= \varphi(V) \sum_{k=n}^{m}\left(\begin{array}{c}
  m \\
  k
  \end{array}\right) * \varphi(K)^{k} *(1-\varphi(K))^{(m-k)}
  $$

  - $\varphi(V)$: Funktionswahrscheinlichkeit des Entscheiders (Voter)



## Failure metrics

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Failure%20Metrics.png" alt="Failure Metrics" style="zoom:80%;" />

- **MTTF** (Mean Time to Failure)
  - mittlere Funktionszeit

  - Einheit, die nicht instand gesetzt wird (Ersatz)

- **MTBF** (Mean Time between Failures) 
  - mittlere Zeit zwischen Ausfällen
  - Einheit, die wieder instand gesetzt wird
  -  Betriebszeit zwischen zwei aufeinanderfolgenden Ausfällen

- **MTTR** (Mean Time to Repair/Recover) 
  
- mittlere Reparaturzeit
  
- Für über die Zeit konstante Ausfallraten gilt außerdem:
  $$
  \lambda = \frac{1}{MTTF}
  $$
  

> Source: https://limblecmms.com/blog/mttr-mtbf-mttf-guide-to-failure-metrics/
>
> - **Mean Time To Repair (MTTR)**
>
>   - refers to the amount of time required to repair a system and restore it to full functionality.
>
>   - MTTR clock **starts ticking when the repairs start and it goes on until operations are restored**. This includes **repair time**, **testing period**, and **return to the normal operating condition**
>
>   - Calculation:
>     $$
>     \mathrm{MTTR}=\frac{\text { total maintenance time }}{\text { total number of repairs }}
>     $$
>
>   - E.g.: a pump that fails three times over the span of a workday. The time spent repairing each of those breakdowns totals one hour. In that case, MTTR would be 1 hour / 3 = 20 minutes.
>
>   - Note: MTTR can also refer to **Mean Time To Recovery**
>
>     - **Mean Time To Recovery** is a measure of the time between the point at which the failure is first discovered until the point at which the equipment returns to operation. So, in addition to repair time, testing period, and return to normal operating condition, it captures **failure notification time and diagnosis**.
>
>       ![Mean time to repair vs Mean time to recovery](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Mean-time-to-repair-vs-Mean-time-to-recovery.jpg)
>
> - **Mean Time Between Failures (MTBF)**
>
>   - measures the predicted time that passes between one previous failure of a mechanical/electrical system to the next failure during normal operation.
>
>   - helps to predict how long an asset can run before the next unplanned breakdwon happens.
>
>     - Note: **the repair time is not included in the calculation of MTBF**.
>
>     ![MTBF (Mean Time Between Failures)](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/MTBF-Mean-Time-Between-Failures.jpg)
>
>   - Calculation:
>     $$
>     \mathrm{MTBF}=\frac{\text { total operational time }}{\text { total number of failures }}
>     $$
>     
>
> - **Mean Time To Failure** 
>
>   - basic measure of reliability **used for non-repairable systems**.
>
>   - represents the length of time that an item is expected to last in operation until it fails (commonly refer to as the lifetime of any product or a device).
>
>     ![Mean time to failure (MTTF)](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Mean-time-to-failure-MTTF.jpg)
>
>   - MTBF Vs. MTTF:
>
>     - **MTBF is used only when referring to repairable items, MTTF is used to refer to non-repairable items**.
>
>   - Calculation
>     $$
>     \mathrm{MTTF}=\frac{\text { total hours of operational }}{\text { total number of units }}
>     $$
>
>     > E.g.: we tested three identical pumps until all of them failed. The first pump system failed after eight hours, the second one failed at ten hours, and the third failed at twelve hours. MTTF in this instance would be (8 + 10 + 12) / 3 = 10 hours.



## Alterung, Ausfall
**Alterungseffekte**

- Vereinfachtes Modell: Konstante Ausfallrate
- Reale Systeme: ariable Ausfallwahrscheinlichkeit über Zeit

**Badewannenkurve**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-23%2011.20.34.png" alt="截屏2020-06-23 11.20.34" style="zoom:80%;" />

- **Frühphase** 

  - Initialausfälle

  - Fertigungsfehler, Bauteildefekte Ausfallrate exponentiell abfallend

- **Betriebsphase**
  * Nahezu konstante Ausfallrate

- **Spätphase** 
  - Alterungseffekte
  - Ausfallrate exponentiell ansteigend

### Ausfallrate

Ausfallrate $z(t)$:
$$
z(t) = \frac{f_L(t)}{R(t)} = \frac{1}{R(t)}\cdot \frac{d}{dt}F_L(t) = \frac{1}{R(t)}\cdot \frac{d}{dt}(1-R(t)) = -\frac{1}{R(t)}\cdot \frac{d}{dt}R(t)
$$
