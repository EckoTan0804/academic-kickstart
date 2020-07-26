---
# Basic info
title: "Verbindungsnetzwerke"
date: 2020-07-24
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
        weight: 4

# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 33
---



## Verbindungsnetzwerke in Multiprozessoren

- Ermöglichen die Kommunikation und Kooperation zwischen den Verarbeitungselementen (Knoten)

  $\Rightarrow$ Zuverlässiger Austausch von Informationen :clap:

- Einsatz eines Verbindungsnetzwerks
  - Chip-Multiprozessor (CMP)
  - Multiprozessor mit verteiltem Speicher (nachrichtenorientierter Multiprozessor)
  - Multiprozessor mit gemeinsamem Speicher

## Charakterisierung von Verbindungsnetzwerken

### Definition

#### Latenz (latency)

- **Übertragungszeit** einer Nachricht $T\_{msg}$

  - die Zeit, die für das Verschicken einer Nachricht von einer bestimmten Länge zwischen zwei Prozessoren benötigt wird

  - Besteht aus

    - **Startzeit $t\_s$ (Message Startup Time)**

      Die Zeit, die benötigt wird, um die Kommunikation zu initiieren

    - **Transferzeit $t\_w$** pro übertragenem Datenwort

      hängt von der physikalischen Bandbreite des Kommunikationsmediums ab.

    - Voraussetzung: Verbindungsnetz ist **konfliktfrei**

- **Software overhead**

- **Kanalverzögerung (channel delay)**

  - Dauer für die Belegung eines Kommunikationskanals durch eine Nachricht
  - **Kanal**: Physikalische Verbindung zwischen Schalterelementen oder Knoten mit einem Puffer zum Halten der Daten während ihrer Übertragung
  - **Verbindung (link)**: Menge von Leitungen

- **Schaltverzögerung, Routing-Verzögerung (switching delay, routing delay)**

  - Zeit, einen Weg zwischen zwei Knoten aufzubauen
  - Pfadberechnung oder Wegefindung (Routing)

- **Blockierungszeit (contention time)**

  - Wird verursacht, wenn zu einem Zeitpunkt mehr als eine Nachricht auf eine Netzwerkressource zugreifen

- **Blockierung (contention)**

  - Ein Verbindungsnetzwerk heißt **blockierungsfrei**, falls jede gewünschte Verbindung zwischen Prozessoren oder zwischen Prozessoren und Speichern *unabhängig von* schon bestehenden Verbindungen hergestellt werden kann

#### Durchsatz oder Übertragungsbandbreite (bandwidth)

- *Maximale* Übertragungsleistung des Verbindungsnetzwerkes oder einzelner Verbindungen
- Meist in **Megabits pro Sekunde (MBit/s)** oder **Megabytes pro Sekunde (MB/s)**

##### Bisektionsbandbreite (bisection bandwidth)

*Maximale* Anzahl von Megabytes pro Sekunde, die das Netzwerk über die Bisektionslinie, die das Netzwerk in zwei gleiche Hälften teilt, transportieren kann

#### Diameter oder Durchmesser $r$ (diameter)

- *Maximale Distanz* für die Kommunikation zweier Prozessoren

  *(also die Anzahl der Verbindungen, die durchlaufen werden müssen)*

- Man spricht auch von der **maximalen Pfadlänge zwischen zwei Knoten**.

#### Verbindungsgrad eines Knotens $P$ (node degree, connectivity)

Anzahl der direkten Verbindungen, die von einem Knoten zu anderen Knoten bestehen.

#### Mittlere Distanz $d\_a$ (average distance) zwischen zwei Knoten

- Anzahl der Links auf dem kürzesten Pfad zwischen zwei Knoten
- $\frac{P}{d\_a}$:  maximale Anzahl neuer Nachrichten, die von jedem Knoten in einem Zyklus in das Netzwerk eingebracht werden können

#### Komplexität oder Kosten

- Kosten für die Implementierung einer Hardware
- Aufwand für das Verbindungsnetz gemessen in der Anzahl und der Art der Schaltelemente und Verbindungsleitungen

#### Erweiterbarkeit

Multiprozessoren können begrenzt, stufenlos oder nur durch Verdopplung der Anzahl der Prozessoren erweiterbar sein.

#### Skalierbarkeit

Fähigkeit, die wesentlichen Eigenschaften des Verbindungsnetzes auch bei beliebiger Erhöhung der Knotenzahl beizubehalten.

#### Ausfallstoleranz oder Redundanz

- Verbindungen zwischen Knoten sind selbst dann noch zu schalten, wenn einzelne Elemente des Netzes (Schaltelemente, Leitungen) ausfallen.
- Ein *fehlertolerantes* Netz muss also zwischen jedem Paar von Knoten mindestens einen *zweiten, redundanten* Weg bereitstellen.

### Verbindungsnetzwerk (IN)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2020.36.03.png" alt="截屏2020-07-24 20.36.03" style="zoom:100%;" />

Verbindet eine Anzahl von Knoten miteinander, so dass Informationen von einem Quellknoten ($S$) zu einem Zielknoten ($D$) verschickt werden können

- **Knoten**

  - Ist über eine **Netzwerkschnittstelle (NI)** mit dem Verbindungsnetzwerk verbunden
  - Bsp: Cache-Modul, Speichermodul, Rechnerknoten, ...

- **Schaltelement (Switch)**

  - Hat eine Menge von Ein- und Ausgängen
  - Setzt eine **Verbindung** zwischen einem Eingang und einem Ausgang auf, um eine Information zu übertragen, solange die Verbindung besteht
  - $n \times n$ Schaltelement: $n$ Eingänge- und $n$ Ausgänge mit Grad $n$, wobei die Ein- und Ausgänge zum Knoten NICHT mitgezählt werden
  - Puffer bereitstellen: Vermeidung von Blockierung an einem Ausgang

- **Link**

  - **Verbindet** einen *Ausgang* eines Schaltelements oder einer Netzwerkschnittstelle mit dem *Eingang* eines anderen Schaltelements oder Netzwerkschnittstelle
  - Besteht aus Leitungen, über die digitale Informationen transportiert werden
    - **Synchrone** Übertragung: Links und Schaltelemente haben *dieselbe* Taktquelle, dessen Zykluszeit vom langsamsten Element bestimmt wird
    - **Asynchrone** Übertragung: die Komponenten haben **unterschiedlichen** Takt und die Synchronisation erfolgt über *Handshake*
  - **Breite**: \#Bits, die parallel in einem Taktzyklus übertragen werden können
  - **Bandbreite**: $\frac{w}{t}$ Bits pro Zeiteinheit ($w$: Breite, $t$: Taktzykluszeit)

- **Nachricht (message)**

  - Informationseinheit, die von der Quelle zum Ziel über das Verbindungsnetzwerk gesendet wird

  - Länge der Nachricht: ein bis zu beliebige Anzahl von Wörtern

  - Arten der Nachrichtenübertragung

    - **Unicast**: ein Knoten schickt eine Anforderungsnachricht an **einen** Zielknoten
    - **Multicast**: Ein Knoten schickt eine Anforderungsnachricht an **mehrere** Knoten
    - **Broadcast**: Ein Knoten schickt eine Anforderungsnachricht an **alle** Knoten

  - Auftrennen in eine Folge von **Paketen fester Länge**

    - erfolgt auf höheren Protokollebenen
    - mit Hilfe einer Kombination von Hardware- und Software-Mechanismen
    - Pakete werden 
      1. in einen Puffer der Netzwerkschnittstelle gespeichert  
      2. anschließend in das Verbindungsnetzwerk geschickt

  - Paket

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2020.57.54.png" alt="截屏2020-07-24 20.57.54" style="zoom:80%;" />

    - **Header**
      - Enthält die Routing Information zum Bestimmen des Wegs vom Quell- zum Zielknoten über eine Menge von Schaltelementen
      - Anforderungs- und Antworttyp (kann auch im Trailer stehen)
    - **Errorcode**
      - Fehlerbehandlung auf Protokollebene des IN
    - **Payload**: Zu übertragende Daten (*irrelevant* für die Verbindung)

### Switching strategy

#### Durchschalte- oder Leitungsvermittlung (circuit switching)

- **Direkte** Verbindung zwischen zwei oder mehreren Knoten eines Netzes
- Die physikalische Verbindung bleibt für die **gesamte Dauer** der Informationsübertragung bestehen.
  - Paket wird **OHNE Unterbrechung** vom Sender zum Empfänger übertragen
  - Paket muss **KEINE Routing-Information** mitführen, da der Weg aufgebaut wird, bevor das Paket versendet wird
  - Vermeidet Routing-Overhead in jedem Schaltelement
  - ‼️ Alle Netzwerkressourcen auf dem Kommunikationspfad sind NICHT für andere Pakete verfügbar, *bis das gesamte Paket am Ziel angekommen ist*

> [**Circuit switching**](https://en.wikipedia.org/wiki/Circuit_switching) 
>
> - is a method of implementing a [telecommunications network](https://en.wikipedia.org/wiki/Telecommunications_network) in which two [network nodes](https://en.wikipedia.org/wiki/Network_nodes) establish a dedicated [communications channel](https://en.wikipedia.org/wiki/Communications_channel) ([circuit](https://en.wikipedia.org/wiki/Telecommunication_circuit)) through the network before the nodes may communicate. 
> - guarantees the full bandwidth of the channel and remains connected for the duration of the [communication session](https://en.wikipedia.org/wiki/Communication_session). 
> - The circuit functions as if the nodes were physically connected as with an electrical circuit. 
>
> The defining example of a circuit-switched network is the early analog [telephone network](https://en.wikipedia.org/wiki/Telephone_network). When a [call](https://en.wikipedia.org/wiki/Telephone_call) is made from one telephone to another, switches within the [telephone exchanges](https://en.wikipedia.org/wiki/Telephone_exchange) create a continuous wire circuit between the two telephones, for as long as the call lasts.

#### Paketvermittlung (packet switching)

![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/350px-Packet_Switching.gif)

- Datenpakete fester Länge oder Nachrichten variabler Länge werden entsprechend einem **Wegefindungsalgorithmus (routing)** vom Absender zum Empfänger geschickt

- Nachrichten mit Adresse und Daten werden durch das Netzwerk verschickt

  - Adresse wird in jedem Knoten (Switch) gelesen und die Nachricht wird zum nächsten Knoten weitergeleitet, bis die Nachricht das Ziel erreicht

  - Die Ressourcen eines Schaltelements werden nur solange belegt wie sie benötigt werden

  - Konflikt: ein Schaltelement ist belegt und der Weg eines anderen Pakets führt zu diesem Schaltelement

    $\rightarrow$ **Flusskontrolle (flow control)**

    - Strategie, die bestimmt, wann ein Paket von einem Schaltelement zu nächsten Schaltelement transportiert werden soll

> **[Packet switching](https://en.wikipedia.org/wiki/Packet_switching)** is a method of grouping data that is transmitted over a digital network into *[packets](https://en.wikipedia.org/wiki/Network_packet)*. Packets are made of a [header](https://en.wikipedia.org/wiki/Header_(computing)) and a [payload](https://en.wikipedia.org/wiki/Payload_(computing)). Data in the header is used by networking hardware to direct the packet to its destination where the payload is extracted and used by [application software](https://en.wikipedia.org/wiki/Application_software). Packet switching is the primary basis for data communications in [computer networks](https://en.wikipedia.org/wiki/Computer_networks) worldwide.

### Latenz- und Bandbreitenmodelle

Bewertung der Entwurfsalternativen für Verbindungsnetzwerke bezüglich der Kommunikationsleistung

#### End-to-end packet latency model

- Betrachtet die Übertragung eines Pakets **vom Sendeknoten zu einem Empfängerknoten** (End-to-end)

- Annahme: das Paket ist bereit zur Übertragung in einem Puffer an der Quelle

- **End-to-end latency**

  - Zeit, die benötigt wird

    - von diesem Zeitpunkt an
    - bis das gesamte Paket über das Verbindungsnetzwerk übertragen worden ist und in einem Puffer am Empfängerknoten abgelegt ist

  - Auf dem *Sendeknoten* wird das Paket zusammengestellt:

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2021.21.00.png" alt="截屏2020-07-24 21.21.00" style="zoom:80%;" />

    $\Rightarrow$ Anzahl der zu übertragenden Bits = $N\_P + N\_E$

#### End-to-end Latency

Übertragungspipeline:

![截屏2020-07-24 22.02.34](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2022.02.34.png)

**End-to-end packet latency = Sender OH + Time of flight + Transmission time + Routing time + Receiver OH**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2022.04.04.png" alt="截屏2020-07-24 22.04.04" style="zoom:80%;" />

- **Sender overhead (SO)**
  - Zeit für das Vorbereiten des Pakets auf der Senderseite durch Hinzufügen des Envelopes und Ablegen des Pakets in den Sendepuffer der NI
  - üblicherweise eine feste Latenz
  - Der relative Einfluss verschwindet mit größeren Paketen 👏

- **Time of flight**

  - Untere Grenze der Zeit, ein Bit vom Sender zum Empfänger zu senden
    - Circuit Switching
      - Festgelegter Weg
      - Ist bestimmt durch durch die logische Übertragungspipeline
    - Packet Switching
      - berücksichtigt auch Schaltzeit, d. h. den Overhead, der durch die Schaltstrategie verursacht wird

- **Transmission time (Übertragungszeit)**

  - Zusätzliche Zeit, die benötigt wird, alle Bits eines Pakets zu übertragen, nachdem das erste Bit beim Empfänger angekommen ist

  - Hängt von der Linkbandbreite $(N\_{Ph} \times f)$ ab

    - $N\_{Ph}$: Anzahl der Bits eines Phits
      - Phit (physical transfer unit): Informationseinheit, die in einem Zyklus auf einem Link übertragen wird
    - Zykluszeit: $\frac{1}{f}$

  - $\Rightarrow$ Für ein Paket: 
    $$
    \text{Transmission time} = \frac{N\_P + N\_E}{N\_{Ph} \times f}
    $$

- **Routing time**

  - Circuit Switching: Zeit, um den Weg aufzusetzen, bevor ein Teil des Pakets übertragen werden kann
  - Packet Switching: Zeit, den Weg in jedem Schaltelement aufzusetzen

- **Switching time**

  - Wird von der [**Switching-Strategie**](#switching-strategy) bestimmt, die vorgibt, wie ein Paket zwischen zwei Schaltelementen übertragen wird

    - **Store-and-forward**: ein Paket wird *vollständig* von einem Schaltelement zum nächsten übertragen bevor es von dort weitergeleitet wird

      > [**Store and forward**](https://en.wikipedia.org/wiki/Store_and_forward) is a [telecommunications](https://en.wikipedia.org/wiki/Telecommunications) technique in which [information](https://en.wikipedia.org/wiki/Information) is sent to an intermediate station where it is kept and sent at a later time to the final destination or to another intermediate station. The intermediate station, or [node](https://en.wikipedia.org/wiki/Node_(networking)) in a [networking](https://en.wikipedia.org/wiki/Computer_network) context, verifies the [integrity](https://en.wikipedia.org/wiki/Data_integrity) of the message before forwarding it.

    - **Cut-through**: Übertragung eines Pakets von Schaltelement zu Schaltelement in *überlappter* Weise

      > **[Cut-through switching](https://en.wikipedia.org/wiki/Cut-through_switching)**, also called **cut-through forwarding**is a method for [packet switching](https://en.wikipedia.org/wiki/Packet_switching) systems, wherein the switch starts forwarding a [frame](https://en.wikipedia.org/wiki/Frame_(networking)) (or [packet](https://en.wikipedia.org/wiki/Network_packet)) before the whole frame has been received, normally as soon as the destination address is processed. Compared to [store and forward](https://en.wikipedia.org/wiki/Store_and_forward), this technique reduces latency through the switch and relies on the destination devices for error handling. Pure cut-through switching is only possible when the speed of the outgoing interface is equal to or greater than the incoming interface speed.

- **Receiver overhead (RO)**

  Ablegen der Verwaltungsinformation und Weiterleiten des Pakets aus dem Empfangspuffer

##### Bsp: Nachricht ist größer als Payload eines Pakets

- Nachricht wird in mehrere Pakete aufgeteilt
- Der Sender setzt ein Paket zusammen und speist es in das Netzwerk ein
- Bei Packet Switching: das letzte Bit verschwindet aus dem Netzwerk, nachdem die Übertragungszeit um ist und die Netzwerkschnittstelle kann ein neues Paket einspeisen

End-to-end latency: 
$$
\begin{aligned}
\text{End-to-end packet latency} = &\text{Sender OH} + \text{Time of flight} + \text{Receiver OH} + \\\\
 &\text{Transmission time} + \text{Routing time} + \\\\
 &(N - 1) \times (\max(\text{Sender OH, Transmission time, Receiver OH}))
\end{aligned}
$$

- $N$ aufeinanderfolgende Pakete einer Nachricht 
- Langsamste Stufe der Pipeline bestimmt den Durchsatz

Effektive Bandbreite:
$$
\text { effektive Bandbreite }=\frac{\text {Paketgröße}}{\max (\text {Sender OH, Empfänger OH, Ubertragungszeit})}
$$

#### Switching Strategy

- Bestimmt, wie ein Weg in einem Verbindungsnetzwerk aufgebaut und ein Paket vom der Quelle zum Ziel übertragen wird
- Modellannahmen
  - Ein Paket hat auf dem Weg von der Quelle zum Ziel $L$ Schaltelemente zu passieren
  - Paket umfasst $N$ Phits
    - ein Phit = der Betrag an Information, der in einem Taktzyklus in der Übertragungspipeline übertragen werden kann

##### Circuit Switching

1. Aufbau des Weges zwischen Quelle und Ziel
2. Das Paket wird übertragen

Routing time

- Zeit, die notwendig ist, ein einzelnes Phit von der Quelle zum Ziel zu senden und wieder zurück, um die Quelle zu informieren, dass der Weg aufgebaut ist ($=L$)
- Die Routing-Entscheidung in einem Schaltelement benötigt $R$ Netzwerkzyklen ($ = L \times R$)

$$
\text {routing time}=L \times R+ \text {time of flight} =L \times R+L=L(R+1)
$$

Daher
$$
\begin{aligned} \text { End-to-end packet latency } &=\text { Sender } \mathrm{OH}+\text { Time of flight }+\text { Transmission time } \\\\ &+\text { Routing time }+\text { Receiver } \mathrm{OH} \\\\ &=\text { Sender } \mathrm{OH}+L+N+L(R+1)+\text { Receiver } \mathrm{OH} \\\\ &=\text { Sender } \mathrm{OH}+L(R+2)+N+\text { Receiver } \mathrm{OH} \end{aligned}
$$

##### Packet Switching

💡 Datenpakete fester Länge oder Nachrichten variabler Länge werden entsprechend einem **Wegefindungsalgorithmus (routing)** vom Absender zum Empfänger geschickt

- Der Weg wird festgelegt, so wie das Paket von der Quelle zum Ziel fortschreitet
- Die Wegeentscheidung wird in **jedem** Schaltelement getroffen, sobald das Paket es erreicht
- Die Ressourcen eines Schaltelements werden **nur solange belegt, wie sie benötigt werden**

###### Store-and-forward

Ein Paket wird **VOLLSTÄNDIG** von Knoten zu Knoten übertragen

- ALLE Teile eines Pakets müssen von einem Knoten empfangen worden sein, bevor ein Teil vom ihm zum nächsten Knoten geleitet wird
- Jeder Knoten enthält einen Puffer zum Aufnehmen des vollständigen Pakets<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2011.26.57.png" alt="截屏2020-07-25 11.26.57" style="zoom:80%;" />

$$
\begin{aligned} \text { End-to-end packet latency } &=\text { Sender } \mathrm{OH}+\text { Time of flight }+\text { Transmission time } \\\\ &+\text { Routing time }+\text { Receiver } \mathrm{OH} \\\\ &=\text { Sender } \mathrm{OH}+L \times N+N+L \times R+\text { Receiver } \mathrm{OH} \\\\ &=\text { Sender } \mathrm{OH}+\mathrm{N}(L+1)+L \times R+\text { Receiver } \mathrm{OH} \end{aligned}
$$

{{% alert note %}} 

Die Übertragungszeit $N$ und die Time of flight $N \times L$ sind propotional zur Paketgröße $N$ (Anzahl der Phits)

D. h. dier Overhead **steigt mit der Paketgröße**

{{% /alert %}}

###### Cut-through

- Kopfteil der Nachricht
  -  enthält die Empfängeradresse
  -  dekodiert in **jedem** Schaltelement bei der Ankunft des Pakets und bestimmt den einzuschlagenden Weg (Routing OH: $R$ Zyklen)
- Blockierung? 
  - Ja: Das gesamte Paket wird aufgehalten
    - **Flow control unit (Flit)** ist der Teil eines Pakets, der bei einer Blockierung aufgehalten wird, bei cut-through ist das gesamte Paket ein Flit
  - Nein: Paket wird durch die Schaltelemente gemäß der Pipeline-Verarbeitung von der Quelle zum Ziel übertragen
- Kopf-Information wird festgehalten bis letztes Phit angekommen ist
  - Nachdem das gesamte Paket von einem Schaltelement weitergeleitet worden ist, wird dieses freigegeben

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2011.54.56.png" alt="截屏2020-07-25 11.54.56" style="zoom:80%;" />
$$
\begin{aligned} \text { End-to-end packet latency } &=\text { Sender } \mathrm{OH}+\text { Time of flight }+\text { Transmission time } \\\\ &+\text { Routing time }+\text { Receiver } \mathrm{OH} \\\\ &=\text { Sender } \mathrm{OH}+L+N+L \times R+\text { Receiver } \mathrm{OH} \end{aligned}
$$

###### Übertragungsmodi: Wormhole-routing-Modus

- KEINE blockierte Übertragungskanäle $\rightarrow$ Identisch mit Cut-through-Modus

- Falls der Kopfteil der Nachricht auf einen Kanal trifft, der gerade belegt ist, wird er **abgeblockt**. 

  $\rightarrow$ Alle nachfolgenden Übertragungseinheiten der Nachricht verharren dann ebenfalls an ihrer augenblicklichen Position, bis die Blockierung aufgehoben ist. Durch das Verharren werden die Puffer nachfolgender Kanäle auch für weitere Nachrichten blockiert.

## Topologie

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-24%2011.38.36.png" alt="截屏2020-07-24 11.38.36" style="zoom:67%;" />

### Statische Verbindungsnetze

- Nach Aufbau des Verbindungsnetzes bleiben die Verbindungen **fest** (*statisch*)

- Gute Leistung für Probleme mit **vorhersagbaren** Kommunikationsmustern zwischen benachbarten Knoten

#### Vollständige Verbindung

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2012.01.26.png" alt="截屏2020-07-25 12.01.26" style="zoom:80%;" />

- Jeder Knoten ist mit jedem anderen Knoten verbunden
- <span style="color:green">Höchste Leistungsfähigkeit</span>
- <span style="color:red">NICHT praktikabel in Parallelrechnern</span>
  - Netzwerkkosten steigen quadratisch mit der Anzahl der Prozessoren 😱

#### Gitterstrukturen

##### 1-dimensionales Gitter (lineares Feld, Kette)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2012.06.49.png" alt="截屏2020-07-25 12.06.49" style="zoom:80%;" />

- Verbindet $N$ Knoten mit $(N-1)$ Verbindungen
- Grad:
  - Endknoten: 1
  - Zwischenknoten: 2 (mit benachbarten Knoten verbunden)
- Diameter (*Maximale Distanz* für die Kommunikation zweier Prozessoren): $r = N - 1$

- <span style="color:green">Disjunkte Bereiche des linearen Netzwerkes können gleichzeitig genutzt werden</span>
- <span style="color:red">Mehrere Schritte notwendig, um eine Nachricht zwischen zwei nicht benachbarte Knoten zu verschicken</span>

##### k-dimensionales Gitter mit N Knoten 

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2012.10.37.png" alt="截屏2020-07-25 12.10.37" style="zoom:80%;" />

#### Ring

Endknoten eines linearen Feldes verbindet sich miteinander 

##### Unidirektionaler Ring mit N Knoten

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2012.11.43.png" alt="截屏2020-07-25 12.11.43" style="zoom:80%;" />

- Nachrichten werden in **einer Richtung** vom Quellknoten zum Zielknoten verschickt
- Diameter $r = N-1$

- <span style="color:red">Bei Ausfall einer Verbindung bricht die Kommunikation zusammen</span> 😭

##### Bidirektionaler Ring mit N Knoten

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2012.13.16.png" alt="截屏2020-07-25 12.13.16" style="zoom:80%;" />

- **symmetrisches** Netzwerk

- Längste Pfad $\leq \frac{N}{2}$
- Bei Ausfall einer Verbindung bricht die Kommunikation noch NICHT zusammen
  - während zwei Ausfälle von Verbindungen das Netzwerk in zwei disjunkte Teilnetzwerke aufteilen

##### Chordaler Ring

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2012.16.34.png" alt="截屏2020-07-25 12.16.34" style="zoom:80%;" />

- Hinzufügen *redundanter* Verbindungen
  - <span style="color:green">erhöht Fehlertoleranzeigenschaft des Verbindungsnetzwerkes </span>
  - Höherer Knotengrad und kleinerer Diameter gegenüber Ring

#### Baum (Tree)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2012.33.54.png" alt="截屏2020-07-25 12.33.54" style="zoom:80%;" />

- Binärer Baum mit $m$-Ebenen
  - Auf Ebene $m$: $N = 2^m - 1$ Knoten
- Diameter: $r = 2(m - 1)$

- Adressierung der Knoten:
  - Die Knotennummer auf Ebene m besteht aus $m$ Bits
  - Wurzelknoten hat die Nummer 1
  - Kindknoten
    - Linke: Hinzufügen einer 0 an die niederwertige Stelle der Adresse des Elternknoten
    - Rechte: Hinzufügen einer 1 an die niederwertige Stelle der Adresse des Elternknoten

- Routing
  - Finde gemeinsamen Elternknoten P von S und D 
  - Gehe von S nach P und von P nach D

##### Fat Tree

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2012.34.49.png" alt="截屏2020-07-25 12.34.49" style="zoom:80%;" />

- Lösung des Blockierungsproblems in Richtung Wurzel
- Kommunikationskanäle werden **größer**, je **näher** man sich der Wurzel nähert

#### Kubus 

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/293px-Hypercube.gif" alt="img" style="zoom:80%;" />

##### K-ärer n-Kubus (Cubes, Würfel)

- Allgemeine Form eines Kubus-Verbindungsnetzwerkes

- Ringe, Gitter, oder Hyperkubi sind **topologisch isomorph** zu einer Familie von K-ären n-Kubus Netzwerken

  - $n$: Dimension
  - $K$: Radius (Anzahl der Knoten, die einen Zyklus in einer Dimension bilden)

  $\Rightarrow$ Anzahl der Knoten: $N = K^n$

- Adressierung der Knoten: n-stellige k-äre Zahl der Form $a\_0, a\_1, \dots, a\_{n-1}$

  - Jede Stelle $0 \leq a\_i > K$ stellt die Position des Knotens in der entsprechenden $i$-ten Dimension dar, mit $0 \leq i \leq n-1$ 

  - Ein Nachbarknoten in der $i$-ten Dimension zu einem Knoten mit Adresse $a\_0, a\_1, \dots, a\_{n-1}$ kanan erreicht werden mit 
    $$
    a\_0, a\_1,\dots, a\_(i \pm 1) \bmod k, \dots, a\_{n-1}
    $$

##### Hyperkubus (Hypercubes)

- Verallgemeinerter Würfel:

  - Ecken eines $n$-dimensionalen Würfels: $N = 2^n$ Prozessoren
  - Kanten: Verbindung

- Komplexität: $\left(N \cdot \log\_{2} N\right) / 2$

- Diameter: $\log\_{2} N$

- Häufigste Verbindungsstruktur bei den nachrichtengekoppelten Multiprozessoren

- <span style="color:red">Problem: Skalierbarkeit</span> 

  - Jede Erweiterung benötigt mindestens die **Verdopplung** der Prozessorenanzahl 🤪

- e-Cube Routing

  - Knotennummern: **Binärzahlen**, dadurch unterscheiden sich benachbarte Knoten in genau einer Stelle, die zudem die Richtung der Verbindung angeben kann (Hamming Distanz)

  - Einfache Wegewahl: die Bits in Start- und Zieladresse werden mittels einer **XOR**-Verbindung verknüpft und das Resultat bestimmt die möglichen Wege.

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2014.31.20.png" alt="截屏2020-07-25 14.31.20" style="zoom:80%;" />

### Dynamische Verbindungsnetzwerke

Geeignet für Anwendungen mit **variablen und nicht regulären Kommunikationsmustern**

#### Bus

- Wird von den am Bus angeschlossenen Prozessoren *gemeinsam* benützt

- Ein Datentransport zu einem Zeitpunkt

- Nachricht von einer Quelle zu jedem Ziel in einem Schritt

- Busbandbreite
  $$
  \text{Busbandbreite} = w * f
  $$

  - $w$: Anzahl der Datenleitungen (Busbreite)
  - $f$: Frequenz
  - ‼️ Die Bandbreite muss mit dem Produkt der Anzahl der Prozessoren und ihrer Geschwindigkeit abgestimmt werden

- Reduzierung des Busverkehrs: Verwendung von Cache-Speichern mit Cache-Kohärenz-Protokollen

- **Split-Phase Busprotokollen**

  - Gibt den Bus nach der Übertragung einer Speichereferenzanforderung wieder frei

  - Wenn der Speicher bereit ist, das Datum zu liefern, fordert dieser den Bus an und schickt die Daten als Antwort

    $\Rightarrow$ Ermöglicht, dass andere Prozessoren in der Zwischenzeit den Bus anfordern können (Voraussetzung: ein verschränkter Speicher ist vorliegt oder Pipelining ist möglich)

#### Kreuzschienenverteiler (Crossbar)

![截屏2020-07-25 18.54.49](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2018.54.49.png)

- **Vollständig** vernetztes Verbindungswerk mit allen möglichen Permutationen der $N$ Einheiten, die über das Netzwerk verbunden werden
- Anforderung an Hardware-Einrichtung: alle möglichen disjunkten Paare von Prozessoren **gleichzeitig und blockierungsfrei** miteinander kommunizieren können.
  - Je zwei beliebige Elemente aus den verschiedenen Mengen können miteinander kommunizieren.
  - $N!$ Permutationen
  - <span style="color:red">An den Kreuzungspunkten sitzen Schaltelemente: hoher Hardware-Aufwand</span> 
  - Kosten: $N^2$ Schaltelemente (bei $N$ Knoten pro Dimension)

##### Schalterelemente (2x2 Kreuzschienenverteiler)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2019.03.07.png" alt="截屏2020-07-25 19.03.07" style="zoom:80%;" />

- bestehen aus Zweierschaltern mit zwei Eingängen und zwei Ausgängen, die 

  - entweder durchschalten oder 
  - die Ein- und Ausgänge überkreuzen 

  können

#### Mehrstufige Verbindungsnetzwerke (Schalternetzwerke, Permutationsnetzwerke)

- Kompromiss zwischen 
  - der niedrigeren Leistungsfähigkeit von Bussen und 
  - hohem Hardware-Aufwand von Kreuzschienenverteilern
- Oft $2 \times 2$ Kreuzschienenverteiler (Schalterelement) als Grundelement

##### Permutationsnetze

$p$ Eingänge des Netzes können gleichzeitig auf $p$ Ausgänge geschaltet werden und somit wird eine Permutation der Eingänge erzeugt.

- **Einstufige Permutationsnetze**
  - enthalten eine *einzelne* Spalte von Zweierschaltern

- **Mehrstufige Permutationsnetze**
  - enthalten *mehrere* solcher Spalten

  - Spalten: Stufen des Permutationsnetzwerkes

- **reguläre Permutationsnetzwerke**
  - $p$ Eingänge, $p$ Ausgänge und $k$ Stufen mit jeweils $p/2$ Zweierschaltern, wobei die Zahl $p$ normalerweise eine Zweierpotenz ist
- **Irreguläre Permutationsnetzwerke**
  - weisen gegenüber der vollen regulären Struktur Lücken auf

##### Permutationen

- **Mischpermutation $M$ (Perfect Shuffle)**
  $$
  M(a\_n, a\_{n-1}, \dots, a\_2, a\_1) = (a\_{n-1}, \dots, a\_2, a\_1, a\_n)
  $$
  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2019.10.35.png" alt="截屏2020-07-25 19.10.35" style="zoom:80%;" />

- **Kreuzpermutation $K$ (Butterfly)**
  $$
  K(a\_n, a\_{n-1}, \dots, a\_2, a\_1) = (a\_1, a\_{n-1}, \dots, a\_2,  a\_n)
  $$
  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2019.11.21.png" alt="截屏2020-07-25 19.11.21" style="zoom:80%;" />

- **Tauschpermutation $T$ (Butterfly)**

  **Negation** des niedrigwertigen Bits
  $$
  T\left(a\_{n}, a\_{n-1}, \ldots a\_{2}, a\_{1}\right)=\left(a\_{n}, a\_{n-1} \ldots a\_{2}, \overline{a\_{1}}\right)
  $$
  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-25%2019.12.45.png" alt="截屏2020-07-25 19.12.45" style="zoom:80%;" />

