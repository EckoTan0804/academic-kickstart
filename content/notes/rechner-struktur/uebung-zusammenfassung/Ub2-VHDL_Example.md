---
# Basic info
title: "Ub2-VHDL Portmapping Example"
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
        weight: 3

weight: 122
---



Source: [Port mapping example](http://telescript.denayer.wenk.be/~kvb/Labo_Digitale_Synthese/vhdl_portmapping_example.pdf)

## Task

Given two modules as follows:

![截屏2020-06-17 15.07.54](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-17%2015.07.54.png)

```vhdl
library IEEE;
use IEEE.std_logic_1164.all;

entity module_a is 
  port (
    clk: in std_logic;
    input_a: in std_logic;
    output_a: out std_logic
  );
end module_a
  
architecture behav of module_a is
begin
  output_a <= input_a;
end behav;
```

![截屏2020-06-17 15.09.57](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-17%2015.09.57.png)

```vhdl
library IEEE;
use IEEE.std_logic_1164.all;

entity module_b is 
  port (
    clk: in std_logic;
    input_b: in std_logic;
    output_b: out std_logic
  );
end module_b
  
architecture behav of module_b is
begin
  output_b <= input_b;
end behav;
```

Portmap them as follows:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-17%2015.12.41.png" alt="截屏2020-06-17 15.12.41" style="zoom:80%;" />



## Step by step

1. **Create empty top_file shell (only define ports)**

![截屏2020-06-17 15.14.05](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-17%2015.14.05.png)

```vhdl
library IEEE;
use IEEE.std_logic_1164.all;

-- Step 1
entity top_file is
  port (
    clk: in std_logic;
    global_input: in std_logic;
    global_output: out std_logic;
  );
end top_file;
  
architecture behav of top_file is
begin
  
  
end behav;
```

**2. Declare the modules**

![截屏2020-06-17 15.21.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-17%2015.21.48.png)

Note: no instantiation of the modules exists yet!

```vhdl
library IEEE;
use IEEE.std_logic_1164.all;

-- Step 1
entity top_file is
  port (
    clk: in std_logic;
    global_input: in std_logic;
    global_output: out std_logic;
  );
end top_file;
  
architecture behav of top_file is
  
-- Step 2
-- (Same as entity in vhd file, only named component)
component module_a
  port (
    clk: in std_logic;
    input_a: in std_logic;
    output_a: out std_logic
  );
end component;
  
component module_b is 
  port (
    clk: in std_logic;
    input_b: in std_logic;
    output_b: out std_logic
  );
end component
  
begin
 
  
  
end behav;
```

**3. Define needed internal signals**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-17%2015.25.12.png" alt="截屏2020-06-17 15.25.12" style="zoom:80%;" />

```vhdl
library IEEE;
use IEEE.std_logic_1164.all;

-- Step 1
entity top_file is
  port (
    clk: in std_logic;
    global_input: in std_logic;
    global_output: out std_logic;
  );
end top_file;
  
architecture behav of top_file is
  
-- Step 2
-- (Same as entity in vhd file, only named component)
component module_a
  port (
    clk: in std_logic;
    input_a: in std_logic;
    output_a: out std_logic
  );
end component;
  
component module_b is 
  port (
    clk: in std_logic;
    input_b: in std_logic;
    output_b: out std_logic
  );
end component
  
-- Step 3
signal int_signal : std_logic;
  
begin
 
  
  
end behav;
```

**4. Instantiate and portmap to connect the modules**

![截屏2020-06-17 15.27.08](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-17%2015.27.08.png)

```vhdl
library IEEE;
use IEEE.std_logic_1164.all;

-- Step 1
entity top_file is
  port (
    clk: in std_logic;
    global_input: in std_logic;
    global_output: out std_logic;
  );
end top_file;
  
architecture behav of top_file is
  
-- Step 2
-- (Same as entity in vhd file, only named component)
component module_a
  port (
    clk: in std_logic;
    input_a: in std_logic;
    output_a: out std_logic
  );
end component;
  
component module_b is 
  port (
    clk: in std_logic;
    input_b: in std_logic;
    output_b: out std_logic
  );
end component
  
-- Step 3
signal int_signal : std_logic;
  
begin
  
  -- Step 4
  module_a_inst: module_a
  port map (
    input_a => global_input;
    clk => clk;
    output_a => int_signal
  );
  
  module_b_inst: module_n
  port map (
    input_b => int_signal;
    clk => clk;
    output_b => global_output
  );
    
end behav;
```

