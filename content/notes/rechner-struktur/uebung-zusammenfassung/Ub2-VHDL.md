---
# Basic info
title: "Ub2-VHDL"
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
share: false  # Show social sharinsg links?
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
        weight: 2
---

{{% alert warning %}} 



In SS20 ist VHDL NICHT in Vorlesungsinhalt und daher NICHT klausurrelevant! 



{{% /alert %}}

**VHDL** = **V**HSIC **HDL** = **V**ery-High-Speed Integrated Circuits **H**ardware **D**escription **L**anguage

Reference:

- [VHDL mini-reference](https://www.ics.uci.edu/~jmoorkan/vhdlref/vhdl.html)

- [VHDL Reference](https://www.ics.uci.edu/~jmoorkan/vhdlref/)

## ENTITY

Definition der Schnittstelle eines Hardwaremoduls (”Leere Hülle“)

- Ein- und Ausgänge (Ports) eines Hardwaremoduls
- Festlegung der Datentypen der Ein- und Ausgänge

Syntax

```vhdl
entity entity_name is
   generic (generic_list);	
   port (port_list);
end entity_name;
```

Bsp

![截屏2020-06-16 17.29.46](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-16%2017.29.46.png)



## ARCHITECTURE

- ”Füllung“ einer Entity mit Inhalt
  - Inhalt bedeutet: Logik bzw. Aufbau der Schaltung
- Mehrere Architekturen können für eine Entity definiert werden
  - Auswahl mithilfe einer CONFIGURATION
- Verschiedene Möglichkeiten:
  - **Funktionale Verhaltensbeschreibung **
  - **Strukturelle Beschreibung** eines Moduls 
  - Mischung beider Beschreibungsarten

### Signale

- Verbingdung zwischen verschiedenen Komponenten

  - speichern (puffern) auch Werte
  - haben einen Datentyp

- Deklaration:

  - Syntax:

    ```vhdl
    signal signal_name : type;
    
    signal signal_name : type := initial_value;
    ```

  - Bsp

    ```vhdl
    signal a, b, out : bit;
    ```

- Wertzuweisung: `<=`

  ```vhdl
  out <= '1';
  ```

  - Achtung: KEINE mehrfachen Zuweisungen!!!
  - erst gültig nach der Abarbeitung aller Operationen des umfassenden Blocks!

### Variablen

- Werte innerhalb eines Prozesses

- haben einen Datentyp

- Deklaration:

  - Syntax: 

    ```vhdl
    variable variable_name : type;
    ```

  - Bsp:

    ```vhdl
    variable state : bit;
    ```

- Wertzuweisung: `:=`

  ```vhdl
  state := not state;
  ```

  - Erfolgt sofort 👏

  - Deklaration mit Standardwert:

    - Syntax:

      ```vhdl
      variable variable_name : type := initial_value;
      ```

    - Bsp:

      ```vhdl
      variable state : bit := '1';
      ```

### Funktionale Verhaltensbeschreibung

- Beschreibung der **Funktionalität** einer Schaltung
- Die interne Struktur wird **abstrahiert**

- In einfachen Fällen: direkte Funktion
- Wird oft "**Behaviour**" genannt
- Hauptbestandteil: **Prozess**
- Funktionales Verhalten durch **nebenläufige Anweisungen**

#### Prozess

- **Sensitivity List**: Signale, bei deren Änderung der Prozess ausgeführt wird

- Mehrere Prozesse werden **gleichzeitig** abgearbeitet

- Rückhalten der Zuweisungen bis Blockende

- Syntax:

  ```vhdl
  optional_label: process (optional sensitivity list)
  	declarations
  begin
  	sequential statements
  end process optional_label;
  ```

- Bsp:

  ```vhdl
  invertieren : process (invert)
    variable state : bit := ’1’;
  begin
    if invert’event then
      state := not state;
    end if;
    out <= state;
  end process invertieren;
  ```

#### Bsp: ENTITY + ARCHITECTURE (Verhaltensbeschreibung)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-16%2018.05.44.png" alt="截屏2020-06-16 18.05.44" style="zoom:80%;" />

### Strukturelle Beschreibung

- Zusammensetzen verschiedener Untermodule zu einer Schaltung

- Beschreibung einer Schaltung durch Aufteilung in verschiedene Untermodule

- Wird oft ”**Structure**“ genannt.

- Verwendung

  1. **Komponentendeklaration**

     - Deklaration der Schnittstelle der Untermodule

       ⇒ Wiederholung der Entity des Untermoduls

     - Durch die Deklaration wird KEIN Untermodul erzeugt!

  2. **Instantiierung** der Untermodule

     - Erzeugung eines oderer mehrerer Untermoduls des

       angegebenen Typs

     - Mehrere Instanzen, d.h. mehrere Untermoduls vom selben Typ möglich

  3. **Verbingdung** der Untermodule durch Signale

     - Angabe der Verbindungen mithilfe der "port map"

#### Syntax

```vhdl
architecture architecture_name of entity_name is
	declarations
begin
	concurrent statements
end architecture_name;
```

#### Bsp

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-16%2021.05.47.png" alt="截屏2020-06-16 21.05.47" style="zoom:80%;" />



## CONFIGURATION

- Auswahl der gewu ̈nschten Architekturbeschreibung fu ̈r eine Entity (d.h. Auswahl des internen Aufbaus)

- Auswahl der zu verwendenden Beschreibungen fu ̈r einzelne Instanzen in struktuellen Beschreibungen

- Auswahl der gewu ̈nschten Architekturbeschreibung fu ̈r einzelne/alle Instanzen einer Entity

- Festlegung des internen Aufbaus bei mehreren Mo ̈glichkeiten 
- Verbindung von Signalen und Ports,. . .



## Tipps

Nehme Ub1, Aufg.2 als Bsp:

- "Erstellen die zugehörige **Schnittstellenbeschreibung**..."

  $\Rightarrow$ Entity deklaration

  - beschreibt Ein-/Ausgabeschnittstelle 
  - Angabe der Schnittstellen über `port` 
  - Richtung und Typ der Ports
  - Pro Modul nur eine Entity

```vhdl
entity counter is
  port (
    clk, rst_n : in std_logic;
    direction : in std_logic;
    enable : in std_logic; -- enable circuit
    select_n : in std_logic; -- read counter value
    value : out std_logic_vector(5 downto 0);
  );
end entity;
```



- "formulieren die entsprechende **Verhaltensbeschreibung** in VHDL..."
  - Implementierung der Funktionalität eines (Teil-)Moduls in einer

    **Architecture**

  - ”Berechnung“ der Ausgangssignalwerte anhand der Eingangssignale und des bisherigen Zustands 

  - Prozesse zur Bündelung

    - Alle Prozesse laufen prinzipiell parallel 

  - Nebenläufige / asynchrone Zuweisungen

```vhdl
architecture arch_counter of counter is
  signal count : unsigned(5 downto 0) := '00000'; 
begin
  
  p_count : process (rst_n, clk),
  begin 
    -- asynchronous reset
    if rst_n = '0' then
      count <= '00000';

    -- counting function, triggered by clock
    elsif (clk'event and clk='1') then

      -- counter enabled?
      if enable = '1' then
        -- counter direction
        if direction = '0' then
          count <= count + 1;
        else
          count <= count - 1;
        end if;
      end if;

    end if;

  end process;
  
  -- output
  value <= std_logic_vector(count) when select_n='0'
  				else (others => 'Z');
  
end arch_counter;
```



## Notes

### Port map

See: [Port map](https://www.hdlworks.com/hdl_corner/vhdl_ref/VHDLContents/PortMap.htm)

Define the interconnection between instances

Syntax:

```vhdl
port map ( [ port_name => ] expression, ... )
```

👍 Example & Tutorials: [Port mapping example](http://telescript.denayer.wenk.be/~kvb/Labo_Digitale_Synthese/vhdl_portmapping_example.pdf)









