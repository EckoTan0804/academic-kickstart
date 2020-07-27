---
# Basic info
title: "Cache"
date: 2020-07-27
draft: false
# type: docs # page type
authors: ["admin"]
tags: ["Computer Structure", "Cache"]
categories: ["Computer Structure"]
toc: true # Show table of contents?

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: "How does cache work in computer structure?"
share: false  # Show social sharing links?
featured: true
lastmod: true

comments: false  # Show comments?
disable_comment: true
commentable: false  # Allow visitors to comment? Supported by the Page, Post, and Docs content types.

editable: false  # Allow visitors to edit the page? Supported by the Page, Post, and Docs content types.

# Optional header image (relative to `static/img/` folder).
header:
  image: ""
  caption: ""
  
---





Immer größer werdende Lücke zwischen Verarbeitungsgeschwindigkeit von Prozessoren und Zugriffsgeschwindigkeit der DRAM-Speicherchips des Hauptspeichers

Ein technologisch einheitlicher Speicher mit kurzer Zugriffszeit und großer Kapazität ist aus Kostengründen i. A. nicht realisierbar。

**🔧 Lösung: Hierarchische Anordnung verschiedener Speicher und Verschiebung der Information zwischen den Schichten (Speicherhierarchie)**

## Speicherhierarchie


zum **Ausgleich** der unterschiedlichen Zugriffszeiten der CPU und des Hauptspeichers.

2 Strategien:

+ **Cache-Speicher**:

  **Kurze Zugriffszeiten** ------> Beschleunigung des Prozessorzugriffs.

+ **Virtueller Speicher**:

  **Vergrößerung** des tatsächlich vorhandenen Hauptspeichers (z. B. bei gleichzeitiger Bearbeitung mehrerer Prozesse)

| Speicherhierarchie                                     |      |
| ------------------------------------------------------ | ---- |
| Register                                               |      |
| on-chip Cache                                          |      |
| secondary level Cache (SRAM)                           |      |
| Arbeitsspeicher (DRAM)                                 |      |
| Sekundärspeicher(Platten, elektronishe Massenspeicher) |      |
| Archivspeicher(Platten, Bänder, optische Platten)      |      |

von Unten nach Oben:

+ Zunehmende Kosten/Byte 

+ Abnehmende Kapazität

+ Abnehmende Zugriffszeit

## Cache Speicher

### Problem : 

+ die Buszykluszeit moderner Prozessoren ist **kuerzer** als die Zykluszeit preiswerter, großer DRAM-Bausteine

  --------> dies zwingt zum **Einfügen von Wartezyklen**

+ SRAM-Bausteine hingegen, die ohne Wartezyklen betrieben
  werden können, sind jedoch klein und teuer. (*存取周期tRC不等于访问时间tAA是SRAM和DRAM的主要区别之一。*)

    --------> nur **kleine** Speicher können so aufgebaut werden

### Lösung des Problems:

**zwischen** den **Prozessor** und den relativ langsamen, aber billigen **Hauptspeicher** aus DRAM-Bausteinen legt man einen **kleinen, schnellen Speicher** aus **SRAM- Bausteinen**, den sogenannten **Cache-Speicher.**

**Prozessor <---> Cache Speicher(besteht aus SRAM)<--> Hauptspeicher** 

**Auf den Cache-Speicher soll der Prozessor fast so schnell wie auf seine Register zugreifen können.**

### 2 Arten (siehe Folien14 s10)

#### 1. On-Chip-Cache: integriert auf dem Prozessorchip

+ **Sehr kurze Zugriffszeiten** (wie die der
  prozessorinternen Register)

+ Aus technologischen Gründen **begrenzte Kapazität**


#### 2. Off-Chip-Cache: prozessorextern


### Nutzen

1. **Verbesserung der Zugriffszeit des Hauptspeichers eines Prozessors** durch einen Cache **zur Vermeidung von Wartezyklen** (CPU-Cache, Befehls- und Daten-Cache).

2. Verbesserung der Zugriffszeit von Plattenspeichern durch einen Cache (**Plattencache**)

### CPU-Cache-Speicher

#### Definition

ein **kleiner**, **schneller** **Pufferspeicher**, in dem **Kopien derjenigen Teile des Hauptsspeichers bereitgehalten werden**, auf die mit hoher Wahrscheinlichkeit von der CPU als nächstes zugegriffen wird.

#### Wieso kommt es zur einer Leistungssteigerung?

Ein CPU-Cache-Speicher bezieht seine Effizienz im wesentlichen **aus der Lokalitätseigenschaft von Programmen (locality of reference)**, 

d.h. es werden bestimmte Speicherzellen bevorzugt und wiederholt angesprochen (z.B. Programmschleifen)

> 程序的局部性原理（locality of  reference）：CPU当前所需要使用的指令或数据在存储器中很可能是在同一地址的附近。

+ **Zeitliche Lokalität**

  Die Information, die **in naher Zukunft angesprochen** wird, ist mit großer Wahrscheinlichkeit **schon früher einmal angesprochen worden** (Schleifen)

+ **Örtliche Lokalität**

  Ein zukünftiger Zugriff wird mit großer Wahrscheinlichkeit **in der Nähe** des bisherigen Zugriffs liegen (Tabellen, Arrays).


#### Funktionsweise

> Cache工作原理：
>
> 当CPU需要数据或指令时, 它首先访问Cache，看看所需要的数据或指令是否存在于Cache中，方法是：
>
> 将CPU提供的数据或指令在内存中存放位置的内存地址 与 Cache中已存放的数据或指令的地址相比价。
>
> - 相等 
>
>   --> 说明可以在Cache中找到所需的数据或指令（cache命中）--> 不需要任何等待状态， Cache直接把信息传给CPU
>
> - 不相等 
>
>   --> CPU所需的数据或指令不在Cache中（未命中），需要到内存中提取:
>
> 存储器控制电路从内存中取出数据或指令传送给CPU，**同时在Cache中拷贝一份副本**。（为了防止CPU以后在访问同一信息时又会出现不命中的情况，从而降低CPU访问速度相对较慢的内存的概率）。
>
>
> 换而言之，Cache命中率越高，系统性能越好 --> 这要求任何时刻cache控制器都要知道cache中存储的是什么指令、数据。  
>
> Cache的命中率取决于下面三个因素：
>
>   + Cache的大小
>
>     容量相对较大的Cache，命中率会相应的提高
>
>   + Cache的组织结构
>
>   + 程序的特性
>
>     遵循局部性原理（locality of reference， Lokalitätseigenschaft von Programmen）的程序在运行时，Cache命中率也会很高。

#### Lseszugriff(siehe Folien14 s16)

Vor jedem Lesezugriff prüft der μP, ob das Datum im Cache steht.

+ Wenn ja: **Treffer (read hit)**

  das Datum kann **ohne Wartezyklen** aus dem Cache entnommen werden. 

+ Wenn nein: **kein Treffer (read miss)**

  das Datum wird **mit Wartezyklen aus dem Arbeitsspeicher gelesen** und **gleichzeitig in den Cache eingefügt.** 

#### Schreibzugriffe

+ Liegt beim Schreiben ein **Cache-Miss (write miss)** vor, wird **das Datum sowohl in den Arbeitsspeicher als auch in den Cache geschrieben**.

+ Liegt beim Schreiben jedoch ein **Cache-Hit (write hit)** vor, d.h. **ein im Cache stehendes Datum wird durch den Prozessor verändert**    

**---> verschiedene Organisationsformen:**

1. **Durchschreibverfahren: (write through policy)**

   Ein Datum wird von der CPU **immer gleichzeitig** in den **Cache-** und in den **Arbeitsspeicher** geschrieben.

   + Vorteil: **garantierte Konsistenz** zwischen Cache- und Arbeitsspeicher.

   + Nachteil: Schreibzugriffe **benötigen immer die langsame Zykluszeit** des Hauptspeichers und belasten den Systembus.

   > 1．直写式（write through）
   >
   > 即CPU在向Cache写入数据的同时，也把数据写入主存以保证Cache和主存中相应单元数据的一致性，其特点是简单可靠，但由于CPU每次更新时都要对主存写入，速度必然受影响。

2. **Gepuffertes Durchschreibverfahren: (buffered write through policy)**

   Zur Milderung des Nachteils beim Durchschreibverfahren wird ein kleiner **Schreib-Puffer** verwendet, der **die zu schreibenden Daten temporär aufnimmt.**

   Diese Daten werden dann **automatisch vom Cache- Controller in den Hauptspeicher übertragen**, **während der Prozessor parallel dazu mit weiteren Operationen fortfährt.**

   > 2．缓写式（post write）
   >
   > 即CPU在更新Cache时不直接更新主存中的数据，而是把更新的数据送入一个缓存器暂存，在适当的时候再把缓存器中的内容写入主存。在这种方式下，CPU不必等待主存写入而造成的时延，在一定程度上提高了速度，但由于缓存器只有有限的容量，只能锁存一次写入的数据，如果是连续写入，CPU仍需要等待。

3. **Rückschreib-Verfahren: (write back policy)**

   Ein Datum wird von der CPU **nur in den Cachespeicher geschrieben** und durch ein **spezielles Bit** (*altered bit, modified bit, dirty bit*) gekennzeichnet.

   **Der Arbeitsspeicher wird nur geändert, wenn ein so gekennzeichnetes Datum aus dem Cache verdrängt wird.**

   + Vorteil:
     auch Schreibzugriffe können **mit der schnellen Cache-Zykluszeit abgewickelt** werden.

   + Nachteil:
     **Konsistenzprobleme** zwischen Cache- und Hauptspeicherspeicher .

   > 3．回写式（write back）
   >
   > 即CPU只向Cache写入，并用标记加以注明，**直到Cache中被写过的块要被进入的信息块取代时，才一次写入主存。**这种方式考虑到写入的往往是中间结果，每次写入主存速度慢而且不必要。其特点是速度快，避免了不必要的冗余写操作，但结构上较复杂。

### Konsistenzprobleme

---

Ebenfalls können andere Systemkomponenten Daten im Hauptspeicher ändern, **während die CPU noch mit den alten Daten im Cachespeicher arbeitet.**

--> **aufwendige Verfahren bei der Cache-Steuerung zur Verhinderung solcher Inkonsistenzen sind erforderlich** 
(z. B. muss die **Cache-Steuerung über jede Datenänderung im Hauptspeicher informiert werden**).


### Begriffe

1. **Hit-Rate**

   Die **Hit-Rate** bezeichnet die **Trefferquote** im Cache:
   $$
   \text{Hit-Rate} = \text{Anzahl Treffer} / \text{Anzahl Zugriffe}
   $$
   

2. **mittlere Zugriffszeit**

   Die **mittlere Zugriffszeit** berechnet sich annähernd wie folgt:
   $$
   t\_{Access} = (\text{Hit-Rate}) * t\_{Hit} + (1 - \text{Hit-Rate}) * t\_{Miss}
   $$

   - $t\_{Hit}$ : Zugriffszeit des Caches
   - $t\_{Miss}$ : Zugriffszeit ohne den Cache

### Aufbau eines Cache-Speichers (Folien14 s27)


#### Grob Struktur

Ein Cache-Speicher besteht aus zwei Speicher-Einheiten:

+ **Datenspeicher**:

  enthält die im Cache abgelegten **Daten**

+ **Adressspeicher**:

  enthält **die Adressen dieser Daten im Arbeitsspeicher**

#### Daten   

+ **Jeder Dateneintrag** besteht aus **einem ganzen Datenblock**(*一行*) (**cache-line**, bis 64 Byte).

+ Mit jedem Datum, auf das der Prozessor zugreift, wird die **Umgebung miteingelagert** *(Hoffnung auf Lokalität von Programmen)*.

+ Im **Adressspeicher** wird **die Basisadresse jedes Blocks** abgelegt

+ Jede Cache-Zeile enthält **ein (Adress-, Daten-)Paar und Statusbits**.

  **eine Cache-Zeile**:

  | Adressspeicher（Tag）                      | Statusbits    | Daten |      |
  | ------------------------------------------ | ------------- | ----- | ---- |
  | Adresse(Adress der Daten in HauptSpeicher) | valid + dirty | Daten |      |

+ Ein **(Daten)-Block** ist eine **zusammengehörende Reihe
  von Speicherzellen (Cache-line).**

+ Dazugehörig wird ein **Adressetikett (Cache-Tag)， enthält die Adresse des aktuellen Blocks im Hauptspeicher，** im
  **Adress-Speicher** ablegt.

+ Die **Statusbits** sagen aus, **ob die Daten im Cache gültig sind.**

#### Komparator

ermittelt, ob das zu einer auf dem Adressbus liegende Adresse gehörende Datum auch **im Cache** abgelegt worden ist, durch **Adressvergleich mit den Adressen im Adressspeicher**

Dieser Adressvergleich muss **sehr schnell gehen (möglichst in einem Taktzyklus)**, da sonst der Cachespeicher effektiv **langsamer** wäre als der Arbeitsspeicher.

### Drei Techniken für den Adressvergleich 􏰂---> 3 Cache-Typen:

#### 1. Voll-Assoziativer Cache(Folien14 s33 & s35)

**werden heute nur für sehr kleine auf dem Chip integrierte Caches mit 32 bis 128 Einträgen verwendet.**

Vollparalleler Vergleich aller Adressen im Adressspeicher in einem einzigen Taktzyklus

+ **Vorteile**:

  + ein Datum kann an beliebiger Stelle im Cache abgelegt werden
  + Optimale Cache-Ausnutzung, völlig freie Wahl der Strategie bei
    Verdrängungen

+ **Nachteile**:

  + **Hoher Hardwareaufwand** (für jede Cache-Zeile ein Vergleicher) 􏰂 
    --> nur für **sehr kleine** Cachespeicher realisierbar

  + Die große Flexibilität der Abbildungsvorschrift erfordert eine weitere Hardware, welche die Ersetzungsstrategie (**welcher Block soll überschrieben werden, wenn der Cache voll ist**) realisiert.

  > 全相连映像方式Cache
  >
  > 任意主存单元的数据或指令可以存放到Cache的任意单元中去，两者之间的对应关系不存在任何限制。
  >
  > + 在Cache中，用于**存放数据或指令的静态存储器SRAM**称为 **内容Cache（Daten Cache，Daten Speicher）**
  > + 用于存放**数据或指令在内存中所在单元的地址**的静态存储器称为 **标识Cache（tag Cache，Tag Speicher）**
  >
  > 假设主存地址是16位，每个存储单元8位（64k * 8 Organisation）。假设内容Cache的容量是 128 Byte， 即有128个单元（128行），每个单元（每行）的宽度为8位；表示Cache（Tag Cache）也应该由128个单元（128 行），为了存放主存单元的地址，Tag Cache每个单元（每行）的宽度应为16位。
  >
  > 当CPU要访问内存时， 它送出的16位地址先与Tag Cache中的128个地址比较。
  >
  >    + 若所需数据或指令的**地址在Tag Cache中**   
  >      --> 命中！  
  >      --> 从内容Cache与之对应的单元（行）中读出数据或指令传给CPU

  >    + 若所需数据或指令的**地址不在Tag Cache中**     
  >      --> 从主存中读出所需的数据或指令传给CPU，同时在Cache中存一份副本（**即将数据或指令写入内容Cache，并将该数据或指令所在的内存单元的地址写入Tag Cache**）
  >
  >    显然，对于全相连映像Cache，Cache中存储的数据越多，命中率越高。但增加Cache容量带来的问题是：每次访问内存都要进行大量的地址比较，既耗时效率也低。

  > 另一方面，若Cache容量太小，如16个单元（行），由于命中率太低，CPU就要频繁的等待操作系统将Cache中的信息换入换出，因为在向Cache中写入新信息之前，必须将Cache中已有的信息保存在主存中。

#### 2. Direct-mapped-Cache（Folien14 s37，38 & 41）


Beim Direct Mapped Cache erhält jede Stelle des Hauptspeichers einen **eindeutigen und festen Platz im Cache**（kommt auf den **Index** an）

+ **Nachteil**

  Ständige Konkurrenz der Blöcke (z. B. 0, 64, 128,...), **obwohl andere Blöcke im Cache frei sein können.**

  > 因为内存中的某一个单元定位对应Cache中的一个Block的地址是由Index（地址的低n位部分决定）， 即 内存中的某一个单元的低n位地址 = Cache中的一个Cache Zeile的地址。 这意味着，只要内存中的两个不同单元的地址的低n位相同，就会定位到同一个Cache Zeile，尽管它们的高位地址并不相同。

+ **Vorteil**

  **Geringer Hardwareaufwand für die Adressierung**, da nur ein Vergleicher für alle Tags benötigt wird. 

+ **Merkmale**

  + **Einfache** Hardware-Realisierung (nur **ein Vergleicher** und **ein Tag-Speicher**)  

  + Der **Zugriff** erfolgt **schnell**, weil das Tag-Feld **parallel** mit dem zugehörigen Block gelesen werden kann

  + Es ist **keine Ersetzungsstrategie erforderlich**, weil die direkte Zuordnung keine Alternativen zulässt

  + Auch wenn an anderer Stelle im Cache noch Platz ist, erfolgt **wegen der direkten Zuordnung eine Ersetzung**

  + Bei einem abwechselnden Zugriff auf Speicherblöcke, deren Adressen den **gleichen Index-Teil** haben, erfolgt **laufendes Überschreiben** des gerade geladenen Blocks. 

> 直接映像Cache与全相连映像Cache完全相反，它只需要做一次地址比较即可确定是否命中。
> 在这种Cache结构中，地址分为两部分：
>
>   + **索引（Index）： 地址的低位部分，直接作为内容Cache的地址，定位到内容Cache的相应单元。**
>
>   + **标识（Tag）： 存储在标识Cache中**。
>
> 例：假设CPU送出的内存地址为16位，A0 - A10为索引 ， A11 - A15为标识。
>
> 现在CPU给出的地址是 E898H （`1110 1000 1001 1000`），则`000 1001 1000`为索引， `11101`为标识。
>
> 1. 索引`000 1001 1000`作为地址，用作在内容Cache和标识Cache中各确定一个单元（行），即这两个单元都是由`000 1001 1000`译码得到的（就是用`000 1001 1000`确定一个Cache Zeile）。这样在内容Cache中，就确定了一个存放数据的单元（行）；同时在Cache中，也确定了一个存放标识的单元（行），**此时并不读出数据**。 
>
> 2. **接着拿根据`000 1001 1000`确定的标识Cache中存放的标识与CPU给出的地址中的标识`11101` 比较**：
>    + 相等 --> 命中！将内容Cache中存放的数据读入CPU。
>
>    + 不相等 --> CPU要等待存储器控制器将地址为 E898H 主存单元的内容读入，同时在Cache复制一份副本（包括标识），供CPU将来使用。
>
> 要注意：索引号`000 1001 1000`只对应这一个Cache单元，但可能有32（2 ^ 5 = 32）
> 种标识，即Tag Cache中的五位标识有可能对应这32个地址中的某一个，从`0000 1000 1001 1000`到`1111 1000 1001 1000`。只要标识不匹配（**即根据索引找到的Tag Cache中的那一行的标识** 与 **CPU给出的地址中的标识**（在这里有32种可能性） *不相等*），就表示Cache未命中。
>
>
> **缺点：**
> 直接映像方式的Cache实质上要求 主存的某一个单元，比如说是A单元（假设地址为 `0000 0000 1001 1000`）只能与Cache中的一个特定单元（行）（地址为 `000 1001 1000`（索引））发生联系，而不能与其他Cache单元发生联系。 
>
> --> 主存的低11位地址（A0 - A10）决定了与它发生关联的那个Cache单元的地址。
>
> 如果此时这个Cache单元正被其他低11位地址与A相同的内存单元所占用，比如说B单元（假设B地址为 `0000 1000 1001 1000`,注意A和B的低11位地址相同（Index相同，Tag不同））。那么即使当时Cache中存在其他空闲的单元，A单元的内容也不能被放到Cache中。
>
> 这就是直接映像Cache的缺点：**尽管地址比较的次数是一次，但 不同的内存单元却肯恩共有相同的Cache索引，不同的Cache标识使得Cache仍未命中，仍需访问主存。**

#### 3. n-way-set-assoziativer Cache

---

####Kompromiss zwischen direct-mapped-Cache und vollassoziativen Cache.

Zum Auffinden eines Datums müssen **alle n Tags mit demselben Index parallel verglichen werden**

--------> der Aufwand **steigt** mit der Zahl n;
für große n nähert sich der Aufwand den voll-assoziativen Caches

+ Verbesserte Trefferrate, da hier eine Auswahl möglich ist (der zu verdrängende Eintrag kann unter n ausgewählt


#### Ersetzungssstrategie(notwendig nur bei voll- oder n-fach satzassoziativer Cachespeicherorganisation)

Ersetzungsstrategie gibt an, welcher Teil des Cachespeichers nach einem Cache-Miss durch eine neu geladene Speicherportion überschrieben wird.

+ **Zyklisch** (der zuerst eingelagerte Eintrag wird auch wieder
  verdrängt, FIFO-Strategie)

+ **LRU-Strategie** (least recently used) der am längsten nicht mehr benutzte Eintrag wird entfernt.

+ **Zufällig** (durch Zufallsgenerator)

Meist wird die sehr einfache Strategie gewählt:
Die am längsten nicht benutzte Speicherportion wird ersetzt (**LRU-Strategie, Least Recently Used**).

> 组相连映像Cache
>
> 组相连映像Cache**是介于全相连映像和直接映像Cache之间**的一种结构。在**直接映像Cache**中，每个索引在Cache只能存放**一个**标识。而在**组相连映像**中，对应每个索引，在Cache中**能够存放的标识数量增加了，从而增加了命中率**。
>
> 例如在2路组相连映像Cache中，每个索引在Cache中能存放**两个**标识，即**在Cache中可以存放两个具有相同索引的内存的单元的内容**（这两个内存单元地址的低位部分（Index-teil）相同，但高位部分（Tag）不同）。
>
> 例：
>
> CPU请求访问地址为 **4518H (`0100 0101 0001 1000`)** 主存单元的内容，在2路组相连映像Cache中，与索引`01 0001 1000`对应的标识可以有2个，Cache电路将2个标识分别与`0100 01`比较：
>
>   + 若其中有一个匹配，则命中 --> 将索引`01 0001 1000`所对应的数据读入CPU中。
>
>   + 若两个标识中任何一个都不是`0100 01`的话 --> 未命中 --> Cache控制器从内存中读入所需内容，同时在Cache中备份一份副本。
>
> 类似的，在4路组相连映像Cache中，假设CPU要访问的地址也是**4518H (`0100 0101 0001 1000`)**，需要把四个标识与 `0100 010` 比较。与2路相比，4路Cache的命中率又提高了50%。
>
> 从例子中可以看出，在组相连映像Cache中，比较的次数与相关联的程度有关。
> **n路组相连映像比较次数为n**
> **组的数目越多，性能越高。**但用作标识Cache的SRAM容量也相应增加了，从而加大了成本。8、16路组相连映像Cache中所增加的成本与提高的命中率相比是不划算的；而且增加组的数目，也增加了Tag的比较次数。**目前绝大多数Cache系统实现的时4路**。
>
> 

### Ursachen für die Fehlzugriffe

---

1. **Erstzugriff (compulsory - obligatorisch)**: 

   Beim **ersten Zugriff** auf einen Cache-Block befindet sich dieser noch nicht im Cache- Speicher und **muss erstmals geladen werden** 

   --> **Kaltstartfehlzugriffe (cold start misses)** oder **Erstbelegungsfehlzugriffe (first reference misses).**

2. **Kapazität (capacity)**: 

   Falls der Cache-Speicher nicht alle benötigten Cache-Blöcke aufnehmen kann, **müssen Cache-Blöcke verdrängt und eventuell später wieder geladen werden**.

3. **Konflikt (conflict) :** 

   treten nur bei **direkt abgebildeten** oder **satzassoziativen Cache-Speichern beschränkter Größe** auf

   ein Cache-Block wird verdrängt und später wieder geladen, falls **zu viele Cache-Blöcke auf denselben Satz abgebildet werden** 

   --> **Kollisionsfehlzugriffe (collision misses)** oder **Interferenzfehlzugriffe (interference misses).**

   

### Erzielbare Cache-Trefferquoten

---

1. **Je größer** der Cachespeicher, **desto größer** die Trefferquote

   > Eine Cache-Trefferquote von circa **94%** kann bei einem
   > **64 kByte** großen Cachespeicher erreicht werden

2. **Getrennte** Daten- und Befehls-Cachespeicher sind bei sehr kleinen Cachespeichergrößen vorteilhaft

3. Bei Cachespeichergrößen **ab 64 KByte** sind **Direct Mapped Cachespeicher** mit ihrer Trefferquote nur **wenig schlechter als Cachespeicher mit Assoziativität 2 oder 4.**



**Voll-assoziative Cachespeicher werden heute nur für sehr kleine auf dem Chip integrierte Caches mit 32 bis 128 Einträgen verwendet.**

**Bei größeren Cachespeichern findet sich zur Zeit ein Trend zur Direct Mapped Organisation oder 2 - 8 fach assoziativer Organisation.**  

### Anbindung des Caches an den Systembus (Folien15 s35) 

---

#### 1. Cache-Controller

#### Cache-Controller =  Tag-RAM + Steuerung + Tag-Komparator

+ **auf einem Chip integriert**(Da dieser sehr schnell sein muß)

+ Cache-Controller **übernimmt die Steuerung der Treiber zum Systembus** (Systembuszugriff **nur bei Cache-Miss**, sonst ist der Systembus für andere Komponenten frei), sowie der Systembussignale zur Einfügung von Wartezyklen bei Cache-Miss (READY, HOLD, HOLDA, ...)


#### 2. Cachespeicher 

selbst ist **seperat** mit **SRAM-Bausteinen aufgebaut.**


### Verwendung mehrerer Caches(Folien15 s34)

---


#### 1. First-Level-Cache (On-Chip-Cache)

Häufig **getrennte** On-Chip-Caches**(Harvard-Architektur**) : 

+ **Befehlscache** 

  für die Befehle 

+ **Datencache** 

  für die Daten.

  **paralleler Zugriff auf Programm und Daten**

**Zusammenfassen:**

+ integriert auf dem Prozessorchip 


+ Sehr **kurze Zugriffszeiten** (**wie** die der prozessorinternen **Register**)


+ Aus technologischen Gründen **begrenzte Kapazität**

#### 2. Secondary- Level-Cache (On-Board-Cache, 64 - 1024 KByte groß)  


+ Außerhalb des Prozessor-Chips(**prozessorextern**)

+ größer als On-Chip-Cache

+ Der Secondary-Level-Cache kann **parallel zum Hauptspeicher an den Bus angeschlosssen werden (Look-Aside-Cache)**. Er sorgt dafür, dass bei einem First-Level-Cache-Miss **die Daten schnell nachgeladen** werden können


### Cache-Kohärenzproblem

---

+ **Gültigkeitsproblem**, das beim Zugriff mehrerer Verarbeitungselemente (z. B. Prozessoren) auf Speicherworte des Hauptspeichers entsteht.

+ **Kohärenz** bedeutet das **korrekte Voranschreiten des Systemzustands** durch ein abgestimmtes Zusammenwirken der Einzelzustände.

+ Im Zusammenhang mit dem Cache muss das System dafür sorgen, dass **immer die aktuellsten Daten und nicht veraltete Daten aus dem Cache gelesen werden**.

**Ein System ist konsistent, wenn alle Kopien eines Datums im Hauptspeicher und den verschiedenen Cachespeichern identisch sind. Dadurch ist auch die Kohärenz sichergestellt, jedoch entsteht ein hoher Aufwand.**

--> konsistent **Datums in Hauptspeicher** und **deren Kopien in Cache** muss **identisch** sein!!!


> 在计算机科学中，缓存一致性（英语：Cache coherence，或cache coherency），又译为缓存连贯性、缓存同调，是指保留在高速缓存中的共享资源，保持数据一致性的机制。

> 在一个系统中，当许多不同的设备共享一个共同存储器资源，在高速缓存中的数据不一致，就会产生问题。这个问题在有数个CPU的多处理机系统中特别容易出现。

> 缓存一致性可以分为三个层级：
>
> + 在进行每个写入运算时都立刻采取措施保证数据一致性
> + 每个独立的运算，假如它造成数据值的改变，所有进程都可以看到一致的改变结果
> + 在每次运算之后，不同的进程可能会看到不同的值（这也就是没有一致性的行为）

<br>
<br>

### Bus-Schnüffeln (Bus-Snooping)

---

In **Mehrprozessorsystemen**, bei denen mehrere Prozessoren mit lokalen Cachespeichern an einen **gemeinsamen Bus/Hauptspeicher** angeschlossen sind, verwendet man das sogenannte **Bus-Schnüffeln**

Die **Schnüffel-Logik** jedes Prozessors **hört am Bus die Adressen mit**, die **die anderen Prozessoren** auf den Bus legen. Die **Adressen auf dem Bus** werden **mit den Adressen, der im Cache gespeicherten** Daten, **verglichen**

Bei **Adressübereinstimmung** am Bus geschieht folgendes:

+ **Schreibzugriff**

  Wenn ein Schreibzugriff auf dieselbe Adresse vorliegt, dann wird der im Cache gespeicherte Cacheblock für **„ungültig“** erklärt **(Write- Invalidate-Verfahren)**, oder mit **aktualisiert (Write-Update- Verfahren).**

+ **Lesezugriff**

  Wenn ein Lesezugriff auf dieselbe Adresse mit einer **modifizierten Datenkopie** im Cachespeicher festgestellt wird, dann legt der Cache- Controller ein **Snoop Status Signal (SSTAT)** auf den Bus.

  1. Der Prozessor, der die Adresse auf den Bus gelegt hat, unterbricht seine Bustransaktion.

  2. Der „schnüffelnde“ Cache-Controller übernimmt den Bus und schreibt den betreffenden Cacheblock in den Hauptspeicher.

  3. Dann wird die ursprüngliche Bustransaktion erneut durchgeführt.

> 个人理解：
>
> 处理器上的Schnüffel-Logik监听着Bus上的别的处理器放上去的地址，Bus上的地址会被拿来跟Cache保存的地址比较
>
> 若Bus中的Adresse和Cache中Tag Speicher中的某个地址相同。
>
> 而且现在多核处理器的某一个处理器对Bus上的这个Adress执行Schreibzugriff 
>
> --> Cache中的这个地址的内存单元的内容将被改变 
>
> --> 造成了：同一个地址，Hauptspeicher中的内容和Cache中的内容不一致（Kohärenz kaputt） 
>
> --> 所以这个地址对应的Cache中的Block（Cache-Zeile）要标上„ungültig“(Write- Invalidate-Verfahren) 或者 aktualisiert (Write-Update- Verfahren) 来表明
> 数据不一致
>
> --> 要访问这个地址的处理器（已经把这个地址放到地址总线Adressbus）终止这次Transaktion
>
> --> Cache Controller接管Bus，把相关的被修改的Block（Cache Zeile）写到内存中相同地址对应的内存单元。
>
> --> 然后刚才被跳过的BusTransaktion重新执行。

## Fragen, die sich ein Speicherhierarchie-Designer stellen muss。


### 1.Block-Abbildungsstrategie

#### Wohin kann ein Block abgebildet werden? 

+ Voll-assoziativ

+ Satz-Assoziativ

+ Direct-Mapped

### 2.Block-Identifikation

#### Wie kann ein Block gefunden werden?

+ Tag/Block

### 3.Block-Ersetzungsstrategie

#### Welcher Block soll bei einem Miss ersetzt werden?

+ Random

+ FIFO

+ LRU


### 4.Schreibe-Strategie 

#### Was passiert bei einem Schreibzugriff?  

+ Durchschreibverfahren(write through policy)

+ Gepuffertes Durchschreibverfahren: (buffered write through policy)

+ Rückschreib-Verfahren: (write back policy)

## Cache-Steuerung (Cache controller)

Cache-Steuerung prüft, ob

+ **Bedingung 1**: 

  Der zur Speicheradresse gehörende Hauptspeicherinhalt als Kopie im Cache steht

+ **Bedingung 2**:

  Dieser Cache-Eintrag durch das Gültigkeits-Bit **(Valid- Bit)** als gültig gekennzeichnet ist

Prüfung führt zu einem **Cache-Treffer** oder zu einem **Fehlzugriff**.

**Cache-Fehlzugriff (Cache-miss)**: eine der beiden Bedingungen ist nicht erfüllt.

+ **Lesezugriffe (read miss)**, Dann:

  1.  Lesen des Datums aus dem Hauptspeicher und Laden des Cache-Speichers
  2.  Kennzeichnen der Cache-Eintrag als **gültig** (V-Bit setzen)
  3.  Speichern der Adressinformation im Adress-Speicher des Cache-Speichers

+ **Schreibzugriffe (write miss)**

  Aktualisierungsstrategie bestimmt, ob

  + der entsprechende Block in den Cache geladen und dann **mit dem zu schreibenden Datum aktualisiert** wird 

  oder

  + nur der Cache aktualisiert wird und **der Hauptspeicher unverändert** bleibt


**3 Strategie :** 

  1. Durchschreibverfahren (write through)

  2. Gepuffertes Schreibverfahren (write buffer)

  3. Rückschreibverfahren (write back)  