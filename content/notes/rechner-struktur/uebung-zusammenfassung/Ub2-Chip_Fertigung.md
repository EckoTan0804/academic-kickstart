---
# Basic info
title: "Ub2-Chip Fertigung"
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
        weight: 1
---


1. Fertigung auf **Wafer**n
   - Größe/Durchmesser des Wafers ⇒ Grundfläche

2. Aufteilen in einzelne Chip-Plättchen (**Die**)
   - Fläche/Form des Dies ⇒ Dies per Wafer

## Kenngrößen

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Chip_Fertigung.png" alt="Chip_Fertigung" style="zoom:80%;" />

**Kenngrößen des Wafers**

- $cost_{\text{wafer}}$: Fertigungskosten des rohen Wafers (Siliziumscheibe)
- $d_{\text{wafer}}$: Größe/Durchmesser
  - liefert Fläche: $A_{\text {wafer}}=\pi\left(d_{\text {wafer}} / 2\right)^{2}$

- $Y_{\text{wafer}}$: Yield/Ausbeute ("gute" wafer)

**Kenngrößen des Dies**

- $A_{\text{die}}$: Die-Fläche 

- **Dies per Wafer (DPW)**
  $$
  \begin{array}{ll}
  DPW &= \text{theoretisches Maximum} - \text{Verschnitt} \\
  &= \frac{A_{\text {water }}}{A_{\text {die }}}-\text{Verschnitt} \\
  &= \frac{\pi \left(d_{\text {water }} / 2\right)^{2}}{A_{\text {die }}}-\frac{\pi  d_{\text {water }}}{\sqrt{2  A_{\text {die }}}}
  \end{array}
  $$

  > DPW = area ratio - circumference / (die diagonal length)

- $DPUA$: Fehlerquote (defects per unit area)
- $\alpha$: Technologiekonstante (Maß für Komplexität bzw. Fertigungstechnologie)

- $Y_{\text{die}}$: Yield/Ausbeute, funktionsfähige Dies
  $$
  Y_{d i e}=Y_{\text {wafer}} \cdot \left(1+\frac{D P U A \cdot A_{\text {die}}}{\alpha}\right)^{-\alpha}
  $$

- Fertigungskosten pro Die

  $$
  \operatorname{cost}\_{d i e}=\frac{cost\_{\text {wafer }}}{D P W * Y\_{d i e}}
  $$

**Test und Assemblierung**

- Kosten

  - Die-Test: $cost_{\text{die-test}}$

  - Packaging: $cost_{\text{packaging}}$
    - Packaging-Kosten beinhalten zusätzliche Test-Kosten (IC-Test, Endkontrolle)

- Endausbeute/Yield: $Y_{\text{final}}$

**Gesamtkosten**

- Pro integriertem Schaltkreis (IC):
  $$
  cost_{I C}=\frac{cost_{\text {die }}+cost_{\text {die-test }}+cost_{\text {packaging }}}{Y_{\text {final }}}
  $$

