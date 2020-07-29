---
# Basic info
title: "Multiprozessoren mit gemeinsamem Speicher"
date: 2020-07-29
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
        parent: multiprozessoren
        weight: 6

# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 35
---

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2010.53.01.png" alt="截屏2020-07-29 10.53.01" style="zoom:80%;" />

- **UMA**: **Uniform Memory Access**
  - Beispiel: **symmetrischer Multiprozessor (SMP)**: *Gleichberechtigter* Zugriff der Prozessoren auf die Betriebsmittel

## Speicherhierarchie

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2010.55.55.png" alt="截屏2020-07-29 10.55.55" style="zoom:80%;" />

- Ausnützen der **Lokalitätseigenschaft** von Programmen 
- **Kompromiss** zwischen Preis und Leistungsfähigkeit
- Speicherkomponenten mit **unterschiedlichen Geschwindigkeiten und Kapazitäten**

### Cache Speicher

- Pufferspeicher mit **schnellem** Zugriff

- Anwendung

  - Pufferspeicher zwischen Hauptspeicher und Prozessor
  - Stellt die während einer Programmausführung jeweils aktuellen Hauptspeicherinhalte für Prozessorzugriffe als **Kopien** möglichst schnell zur Verfügung.

- Aktualisierungsstrategie

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2010.58.40.png" alt="截屏2020-07-29 10.58.40" style="zoom:80%;" />

  - Write-through with no-write allocation

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/800px-Write-through_with_no-write-allocation.svg.png" alt="img" style="zoom: 67%;" />

  - Write-back with write allocation

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/800px-Write-back_with_write-allocation.svg.png" alt="img" style="zoom: 80%;" />

## Gültigkeitsproblem

- wenn diese Prozessoren jeweils unabhängig voneinander auf Speicherwörter des Hauptspeichers zugreifen können.
- Mehrere Kopien des gleichen Speicherwortes müssen miteinander in Einklang gebracht werden.

Eine Cache-Speicherverwaltung heißt **cache-kohärent**, wenn ein Lesezugriff immer den Wert des zeitlich letzten Schreibzugriffs auf das entsprechende Speicherwort liefert.

## Cache-Kohärenz-Problem

### 1.Fall: I/O-Problem

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29 11.06.35.png" alt="截屏2020-07-29 11.06.35" style="zoom:80%;" />

System mit einem Mikroprozessoren und weiteren Komponenten mit Master-Funktion (ohne Cache)

- Zusätzlicher Master (z.B. DMA Controller) kann Kontrolle über Bus übernehmen
  - Kann damit **unabhängig** vom Prozessor auf Hauptspeicher zugreifen 
  - Mikroprozessor und Master teilen sich **gemeinsamen** Datenbereich

Mikroprozessorsystem mit DMA-Controller: <span style="color:red">Zugriff auf **veraltete Daten (stale data)** </span>

- Problem beim Write-Through-Verfahren

  - Situation

    1. DMA-Controller beschreibt eine Speicherzelle, deren Inhalt im Cache als gültig eingetragen war
    2. Der Prozessor führt danach einen Lesezugriff mit der Adresse dieser Speicherzelle durch

    $\to$  <span style="color:red">Prozessor liest veraltetes Datum</span>

  - 🔧 Lösung: **Non-Cachable Data**
    - der vom Prozessor und dem zusätzlichen Master gemeinsam benutzte Speicherbereich wird von der Speicherung im Cache **ausgeschlossen**
    - Aufgabe der Speicherverwaltung
      - Dieser Adressbereich als ***„non-cacheable“*** gekennzeichnet
      - Die Cache-Steuerung wird bei Zugriffen auf den so gekennzeichneten Bereich NICHT aktiv.
      - Es werden auch die für Schnittstellen und Controller reservierten Adressbereiche als ***„non-cacheable“*** gekennzeichnet, um den direkten Zugriff auf deren Daten-; Steuer- und Statusregister zu gewährleisten.

- Problem beim Copy-Back-Verfahren

  - Situation

    1. Der Prozessor führt Schreibzugriff mit der Adresse aus dem gemeinsamen Bereich aus und aktualisiert nur Cache
    2. Der DMA-Controller liest anschließend die Speicherzelle mit dieser Adresse

    $\to$  <span style="color:red">Der DMA-Controller liest veraltetes Datum (im Hauptspeicher)</span>

  - 🔧 Lösung: **Cache-Clear, Cache-Flush**

    - Die Zugriffe von Prozessor und DMA-Controller auf den gemeinsamen Datenbereich werden von **zwei unterschiedlichen Tasks** ausgeführt;

    - In diesem Fall kann die Task, die den DMA-Vorgang auslöst, dafür sorgen, dass der Cache gelöscht wird (d.h nachfolgende Prozessorzugriffe führen zu einem Neuladen des Cache)

    - **Write-Through: Cache-Clear**

      Die Cache-Einträge werden auf *ungültig* gesetzt.

    - **Copy-Back: Cache-Flush**

      Alle mit „dirty“ gekennzeichneten Einträge im Cache werden in den Hauptspeicher zurückgeschrieben, danach werden Cache-Einträge auf ungültig gesetzt.

### 2. Fall: Speichergekoppeltes Multiprozessorsystem

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2012.00.41.png" alt="截屏2020-07-29 12.00.41" style="zoom:80%;" />

Mehrere Prozessoren mit **jeweils eigenen Cache-Speichern** sind über einem Systembus an einen gemeinsamen Hauptspeicher angebunden.



## Cache-Kohärenz und Konsistenz

### Vereinfachte und intuitive Definition

- Ein Speichersystem ist **kohärent**, wenn jeder Lesezugriff auf ein Datum den aktuell geschriebenen Wert dieses Datums liefert

- **Kohärenz**: definiert, *welcher* Wert bei einem Lesezugriff geliefert wird
- **Konsistenz**: bestimmt, *wann* ein geschriebener Wert bei einem Lesezugriff geliefert wird

### Kohärenz

Ein Speichersystem ist **kohärent**, wenn

- Einhaltung der Programmordnung

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2012.31.30.png" alt="截屏2020-07-29 12.31.30" style="zoom:80%;" />

  Ein Lesezugriff eines Prozessors P auf eine Speicherstelle X, der einem Schreibzugriff von P auf die Stelle X folgt und KEINE Schreibzugriffe anderer Prozessoren zwischen dem Schreiben und dem Lesen von P stattfinden, liefert immer den Wert, den P geschrieben hat.

- Kohärente Sicht des Speichers

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2012.33.57.png" alt="截屏2020-07-29 12.33.57" style="zoom:80%;" />

  Ein Lesezugriff eines Prozessors P auf eine Speicherstelle X, der auf einen Schreibzugriff eines anderen Prozessors auf die Stelle X folgt, liefert den geschriebenen Wert, falls der Lese- und Schreibzugriff **zeitlich ausreichend getrennt erfolgen und in der Zwischenzeit keine anderen Schreibzugriffe auf die Stelle X erfolgen**.

- Write Serialization

  Schreibzugriffe auf die eine Speicherzelle serialisiert werden; d.h. zwei Schreibzugriffe auf eine Speicherstelle durch zwei Prozessoren werden durch die anderen Prozessoren in der selben Reihenfolge gesehen.

### Konsistenz

Frage: Wann wird ein geschriebener Wert sichtbar?

- Man kann NICHT fordern, dass ein Lesezugriff auf eine Stelle X *sofort* den Wert liefert, der von einem Schreibzugriff auf X eines anderen Prozessors stammt
- **Konsistenzmodell**: Strategie, wann ein Prozessor die Schreiboperationen eines anderen Prozessors sieht



## Kohärenz-Protokolle

Ein paralleles Programm, das auf einem Multiprozessor läuft, kann mehrere Kopien eines Datums in mehreren Caches haben

- **Migration** bei kohärenten Caches
  - Daten können zu einem lokalen Cache migrieren und dort in einer transparenten Weise verwendet werden
  - Reduziert die Latenz für einen Zugriff auf ein gemeinsames Datum, das auf einem entfernten Speicher liegt :clap:
  - Reduziert auch die erforderliche Bandbreite auf den gemeinsamen Speicher :clap:
- **Replikation** bei kohärenten Caches
  - Gemeinsame Daten können in als Kopien in lokalen Caches vorliegen, wenn beispielsweise diese Daten gleichzeitig gelesen werden
  - Reduziert die Latenz der Zugriffe und die Möglichkeit einer Blockierung beim Zugriff auf das gemeinsame Datum :clap:

### Write-invalidate & Write-update

- **Write-invalidate-Protokoll**
  - Sicherstellen, dass ein Prozessor *exklusiven Zugriff* auf ein Datum hat, bevor er schreiben darf
  - Vor dem Verändern einer Kopie in einem Cache-Speicher müssen alle Kopien in anderen Cache-Speichern für „*ungültig*“ erklärt werden ($\to$ "invalidate")

- **Write-update-Protokoll**
  - Beim Verändern einer Kopie in einem Cache-Speicher müssen alle Kopien in anderen Cache-Speichern *ebenfalls* *verändert* werden, wobei die Aktualisierung auch verzögert (spätestens beim Zugriff) erfolgen kann

- Vergleich: 

  - Mehrfaches Schreiben auf eine Stelle OHNE dazwischen auftauchende Lesezugriffe

    - Write-Update: erfordert mehrere Broadcast-Schreiboperationen
    - Write-Invalidate: Nur eine Invalidierung

  - Cache-Zeilen mit mehreren Wörtern

    - Write-Update

      - Arbeitet auf Wörtern

      - Für jedes Wort in einem Block, das geschrieben wurde, ist ein Write- Broadcast notwendig

    - Write-Invalidate

      - Die erste Schreiboperation auf ein Wort eines Cache-Blocks erfordert eine Invalidierung

### Hardware-Lösung

- **Tabellen-basierte Protokolle (directory-based protocols)**

  Der Zustand eines Blocks im physikalischen Speicher wird in einer Tabelle (directory) festgehalten

- **Snooping-Protokolle (Bus-Schnüffeln)**
  - Jeder Cache, der eine Kopie der Daten eines Blocks des physikalischen Speichers enthält, hat ebenso eine Kopie des Zustands, in dem sich der Block befindet
  - KEIN zentraler Zustand wird festgehalten
  - Caches sind an einem gemeinsamen Bus und alle Cache-Controller beobachten (oder schnüffeln) am Bus, um bestimmen zu können, ob sie eine Kopie eines Blocks enthalten, der benötigt wird

### MESI-Kohärenzprotokoll

- Jeder Cache verfügt über **Snoop-Logik und Steuersignale**

  - **Invalidate-Signal**

    Invalidieren von Einträgen in den Caches anderer Prozessoren.

  - **Shared-Signal**

    Anzeige, ob ein zu ladender Block bereits als Kopie vorhanden ist.

  - **Retry-Signal**

    Aufforderung für einen Prozessor, das Laden eines Blockes abzubrechen. Das Laden wird dann wieder aufgenommen, wenn ein anderer Prozessor aus dem Cache in den Hauptspeicher zurück geschrieben hat.

- Jede Cache-Zeile ist um zwei **Statusbits** erweitert,  um die Protokollzustände anzuzeigen
  - **Invalid (I)**

    Die betrachtete Cache-Zeile ist *ungültig*

    - Lese- und Schreibzugriff auf diese Zeile veranlassen die Cache-Steuerung, den Speicherblock in die Cache-Zeile zu laden.
    - Die anderen Cache-Steuerungen, die den Bus beobachten, zeigen mit Hilfe des Shared-Signals an, ob dieser Block *gespeichert ist (Shared Read Miss)* oder *nicht (Exclusive Read Miss).*

  - **Shared (S)**

    Shared Unmodified: der Speicherblock existiert als Kopie in der Zeile des betrachteten Caches sowie gegebenenfalls in anderen Caches.

    - Lesezugriff auf die Cache-Zeile (Read-Hit): 
      - Der Zustand wird nicht verändert.
    - Schreibzugriff auf die Cache-Zeile (Write-Hit): 
      - Die Cache-Zeile wird geändert und geht in den Zustand M über.
    - Ausgeben des Invalidate-Signals, woraufhin die Caches, bei denen diese Cache-Zeile ebenfalls im Zustand S ist, diese als ungültig kennzeichnen (Zustand **I**).

  - **Exclusive (E)** 

    Exclusive Unmodified: Der Speicherblock existiert als Kopie nur in der Zeile des betrachteten Caches.

    - Der Prozessor kann lesend und schreiben zugreifen, OHNE den Bus benützen zu müssen.
    - Schreibzugriff:
      - Wechseln in den Zustand M. 
      - Andere Caches sind nicht betroffen.

  - **Modified (M)**

    Exclusive Modified: Der Speicherblock existiert als Kopie NUR in der Zeile des betrachteten Caches. Er wurde nach dem Laden verändert.

    - Der Prozessor kann lesend und schreibend zugreifen, OHNE den Bus benützen zu müssen.

    - Bei einem Lese- oder Schreibzugriff eines anderen Prozessors auf diesen Block (Snoop-Hit) muss dieser in den Hauptspeicher zurückkopiert werden.

      - Snoop-Hit on a Read: M $\to$ S

      - Snoop-Hit on a Write or Read with Intend to Modify: M $\to$ I

    - Der Prozessor, der diesen Block aus dem Hauptspeicher holen will, wird mit Hilfe des Retry-Signals darüber informiert, dass zunächst ein *Zurückschreiben* erforderlich ist.

#### Zustandsgraph (lokale Lese- und Schreibzugriffe)

Zustandsübergänge durch lokale Lese- und Schreibzugriffe (d.h. Zugriffe des Prozessors)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2015.25.26.png" alt="截屏2020-07-29 15.25.26" style="zoom:80%;" />

#### Zustandsgraph (außene Lese- und Schreibzugriffe)

Zustandsübergänge , die sich durch Beeinflussung von außen, von Seiten des Busses ergeben. Steuerung durch Snoop-Logik

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2016.25.48.png" alt="截屏2020-07-29 16.25.48" style="zoom:80%;" />

#### Wirkungsweise

Bsp: ein Mikroprozessorsystem mit 2 Prozessoren

- Vier aufeinander folgende Zugriffe auf ein und denselben Speicherblock

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2016.30.23.png" alt="截屏2020-07-29 16.30.23" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2016.30.59.png" alt="截屏2020-07-29 16.30.59" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2016.31.06.png" alt="截屏2020-07-29 16.31.06" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2016.31.36.png" alt="截屏2020-07-29 16.31.36" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2016.31.55.png" alt="截屏2020-07-29 16.31.55" style="zoom:80%;" />

### Multiprozessor mit verteiltem gemeinsamem Speicher, Distributed Shared Memory (DSM)

![截屏2020-07-29 16.33.02](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2016.33.02.png)

- KEINE Möglichkeit, die Broadcast-Eigenschaft des Busses zu nutzen 🤪

- Verzeichnisbasierte (tabellenbasierte) Cache-Kohärenzprotokolle **(directory based)**



### Speicherkonsistenz

**Wichtige Fragen**:

- **Wann** muss ein Prozessor den Wert sehen, den ein anderer Prozessor aktualisiert hat?

- In **welcher Reihenfolge** muss ein Prozessor die Schreiboperationen eines anderen Prozessors beobachten?

- **Welche Bedingungen** zwischen Lese- und Schreiboperationen auf verschiedene Speicherstellen durch verschiedene Prozessoren müssen gelten?

### Speicherkonsistenzmodelle

Spezifizieren die Reihenfolge, in der Speicherzugriffe eines Prozesses von anderen Prozessen gesehen werden

#### Sequentielle Konsistenz

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-29%2016.41.44.png" alt="截屏2020-07-29 16.41.44" style="zoom:80%;" />

- Ein Multiprozessorsystem heißt **sequentiell konsistent**, wenn das Ergebnis einer beliebigen Berechnung *dasselbe* ist, *als wenn die Operationen aller Prozessoren auf einem Einprozessorsystem in einer sequentiellen Ordnung ausgeführt würden.* Dabei ist die Ordnung der Operationen der Prozessoren die des jeweiligen Programms.
- Alle Lese- und Schreibzugriffe werden in einer beliebigen sequentiellen Reihenfolge, die jedoch mit den jeweiligen Programmordnungen konform ist, am Speicher wirksam.
- Entspricht einer überlappenden sequentiellen Ausführung sequentieller Operationsfolgen anstelle einer parallelen Ausführung
- Schreibzugriffe müssen **atomisch** sein, d. h. der jeweilige Wert muss überall gleichzeitig wirksam sein
- Nachteile
  - Führt zu <span style="color:red">sehr starken Einbußen bzgl. Implementierung und damit der Leistung</span>
  - Verbietet vorgezogene Ladeoperationen, nichtblockierende Caches

#### Abgeschwächte Konsistenzmodelle

- Konsistenz NUR zum Zeitpunkt einer Synchronisationsoperation
- Lese- und Schreiboperationen der parallel arbeitenden Prozessoren auf den gemeinsamen Speicher zwischen den Synchronisationszeitpunkten können in *beliebiger* geschehen.
- Konkurrierende Zugriffe auf gemeinsame Daten werden durch **geeignete Synchronisationen** geschützt
- 💡Idee
  - Die Konsistenz des Speicherzugriffs wird nicht mehr zu allen Zeiten gewährleistet, sondern zu **bestimmten, vom Programmierer in das Programm eingesetzten Synchronisationspunkten**
  - Kritische Bereiche
    - Innerhalb dieser Bereiche: Inkonsistenz der gemeinsamen Daten zugelassen
    - Voraussetzung: konkurrierende Lese-/Schreibzugriffe sind durch den kritischen Bereiche unterbunden
    - Synchronisationspunkte: die Ein-/ und Austrittpunkte der kritischen Bereiche
- Bedingungen
  - Bevor ein Schreib- oder Lesezugriff bezüglich irgendeines anderen Prozessors ausgeführt werden darf, müssen ALLE vorhergehenden Synchronisationspunkte erreicht worden sein
  - Bevor eine Synchronisation bezüglich irgendeines anderen Prozessors ausgeführt werden darf, müssen ALLE vorhergehenden Schreib- oder Lesezugriffe ausgeführt worden sein.
  - Synchronisationspunkte müssen **sequentiell konsistent** sein

- Auswrikung

  Synchronisationsbefehle stellen Hürden dar, die von keinem Lese- oder Schreibzugriff übersprungen werden

- Voraussetzung für die Implementierung der schwachen Konsistenz

  hardware- und softwaremäßige Unterscheidung der Synchronisationsbefehle von den Lade- und Speicherbefehlen und eine sequentiell konsistente Implementierung der Synchronisationsbefehle