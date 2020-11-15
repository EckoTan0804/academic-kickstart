---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 1

# Basic metadata
title: "Clean Code"
date: 2020-11-06
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Software Engineering", "Lecture"]
categories: ["Software Engineering"]
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
    software-engineering:
        parent: implementing-high-quality-systems
        weight: 1

---

## Motivation

**Readability** of code is important

- Code is much more often read than written
- Your write code for the next human to read it, not for the compiler/interpreter/computer!

## Object-Oriented Design (OOD)

> A design strategy to build a system “made up of interacting objects that maintain their own local state and provide operations on that state information.” 
>
> [Sommerville]

**SOLID** principles: Five principles of good OO design 

- **S**ingle Responsibility Principle (SRP) 
- **O**pen Closed Principle (OCP)
- **L**iskov Substitution Principle (LSP) 
- **I**nterface Segregation Principle (ISP) 
- **D**ependency Inversion Principle (DIP)

### Single Responsibility Principle (SRP)

> “There should never be more than one reason for a class to change.“
> — R. Martin

- Each responsibility deals with **one core concern**
  - It may also deal with further (cross-cutting) concerns

- Bad smell: Big class (~ >200 LOC, >15 methods/fields) 
  - Useful refactoring: Extract class
- Benefits:
  - Code is easier to understand
  - Adding/modifying functionality should affect few classes 
  - Risk of breaking code is minimised

####  Insertion: Command-Query-Separation

- Separate commands (actions) from simple queries (requests)
- Reason
  - Commands are expected to have side effects on an object’s state
  - Queries should not change the state of an object
  - Appropriate designs are simpler to understand and easier to test

### Open Closed Principle (OCP)

> “Software entities (classes, modules, functions, etc.) should be open for extension, but closed for modifi-cation.”
> — R. Martin, paraphrasing B. Meyer

- 💡 Idea: Modify behaviour by adding new code, NOT by changing old code

- Strongly related to the “Information Hiding Principle”

- Example: Drawing a list of shapes using a switch statement

  ```java
  for (Shape shape : ShapeList) 
    switch (shape.getType()) {
      case SQUARE: square.draw()
      case CIRCLE: circle.draw() 
    }
  ```

  Needs to be modified for new shapes 🤪

  Solution: use abstractions to keep the function open for extension

  ```java
  for (Shape shape : ShapeList) 
    shape.draw();
  ```

### Liskov Substitution Principle (LSP)

> “Functions that use pointers or references to base classes must be able to **use** objects of **derived classes without knowing** it.” 
>
> *— R. Martin*

#### Example

- Square **is-a** Rectangle? Only in a mathematical sense! 

- Square **can-NOT-substitute** Rectangle, because it offers limited behaviour (`setWidth` and `setHeight` are dependent)

  ![截屏2020-11-15 14.38.09](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2014.38.09.png)

LSP is related to B. Meyer‘s **Design by Contract** (DbC):

> *“When redefining a routine [in a derivative], you may only replace its* **precondition** *by a* **weaker** *one, and its* **postcondition** *by a* **stronger** *one.” *
>
> *— B. Meyer*

- In our case, rectangle's `setWidth` postcondition: `width = w` and `height = h`
- Square's `setWidth` postcondition: `width = w` and `height = w`
- Only weaker preconditions and stronger postconditions are allowed, as only they preserve substitutability. It is not allowed to change conditions to *arbitrarily different* ones

Possible solution according to Liskov:

- Square/Rectangle **can-substitute** Shape, 

- if Shape collects

  - less specific behaviour 

    ![截屏2020-11-15 14.43.10](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2014.43.10.png)

  - Alternative: Drop `height = h` from Rectangle’s postcondition

### Interface Segregation Principle (ISP)

> “Clients should not be forced to depend upon interfaces that they do not use.” 
>
> *— R. Martin*

Interfaces should be kept as lean as possible

- **High cohesion**: Interfaces should only be concerned with single concepts
- **Interface pollution**: Interfaces should not depend on other interfaces just because a subclass requires those 
- Interfaces should be separated if used by different clients
- Refactorings: Extract interface/superclass

Example: 

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2014.48.06.png" alt="截屏2020-11-15 14.48.06" style="zoom:80%;" />

### Dependency Inversion Principle (DIP)

> “**A.** High level modules should not depend upon low level modules. Both should **depend upon abstractions**.
>
> **B.** Abstractions should not depend upon details. Details should depend upon abstractions.”
>
> *— R. Martin*

Example:

![截屏2020-11-15 14.50.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2014.50.48.png)

Better design:

![截屏2020-11-15 14.51.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2014.51.23.png)

#### **Why “Inversion”?**

- An interface has been used to *invert* the dependency between packages

- But in general: Add abstract concept that both classes A and B depend on

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2014.53.02.png" alt="截屏2020-11-15 14.53.02" style="zoom:80%;" />

## More Principles

### Law of Demeter (don’t talk to strangers)

> A module should not know about the innards of the objects it manipulates.

- Corresponds to the bad smell “Message Chains”:

  ```java
  value = getClassA().getClassB(). ... .getNeededValue();
  ```

  👆  Ties code to particular class structure, which is likely to break. :cry

- Rule: A method `m` of a class `C` should only call the methods of

  - `C`

  - An object created by `m`

  - An object passed as an argument to `m`

  - An object held in an instance variable of `C`

Example:

- Violation

  ```java
  class Motor {
      
    	public void startEngine() {
          // start the motor
      } 
  }
  ```

  ```java
  class Car {
      
      public Motor motor;
      
      public Car() {
          motor = new Motor();
      }
  
  }
  ```

  ```java
  class Driver {
      
      public void drive() {
        	Car myCar = new Car();
        	myCar.motor.startEngine(); // violation!!!
      }
  }
  ```

- Solution

  ```java
  class Car {
      private Motor motor;
      
      public Car() {
          motor = new Motor();
      }
      
      public void getReadyToDrive() {
          this.motor.startEngine()
      }
  }
  ```

  ```java
  class Drive {
      public void drive() {
          Car myCar = new Car();
          myCar.getReadyToDrive();
      }
  }
  ```

### Boy Scout Rule 

> „Leave the campground cleaner than you found it!“
>  *— The Boy Scouts of America*

- Code degrades as time passes 

- We seldom start with a greenfield

- *Being honest:* 

  - *To the code*

  - *To your colleagues*
  - *To yourself about the code*

- *Refactor your code before checking it in*

### Principle of Least Surprise

> Any function or class should implement the behaviours that another programmer could reasonably expect
>
> Also called [**principle of least astonishment** (**POLA**)](https://en.wikipedia.org/wiki/Principle_of_least_astonishment)
>
> "If a necessary feature has a high astonishment factor, it may be necessary to redesign the feature."

- If **obvious behaviour** remains unimplemented, readers and users... 
  - no longer depend on their **intuition** about function names
  - fall back on reading internals

### Coding Conventions

#### Naming

- Standardised (with respect to a project or team) 

- Meaningful, i.e. clear for everyone 

- **Intention-revealing**:

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2015.36.36.png" alt="截屏2020-11-15 15.36.36" style="zoom:67%;" />

- Make meaningful distinction and avoid disinformation
  - Hints on **context**
  - Hints on **types**
  - Certain **prefixes**
- Avoid **noninformation** 
  
  - Except for well-accepted cases (e.g. `i` as a loop counter)

#### Commenting

> “Don’t comment bad code—rewrite it.“
>
> *— B. W. Kernighan, P. J. Plaugher*

Good comments are

- **explaining**
  - Legal issues
  - Performance issues
  - Train of thought
  - Intent
  - Algorithms
- Good comments are **warning**
  - Of consequences
  - Over importance
- Good comments are **informative**
  - Open issues, to-dos

Whenever possible, use **well-named code** to tell what is done 

- Intermediate variables explaining steps

- Extra methods encapsulating expressions

  ![截屏2020-11-15 15.44.54](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2015.44.54.png)

#### **Formatting**

- Visually representing levels of cohesion
- **Vertical** openness between concepts, 
  - e.g. declarations 
  - e.g. add blank lines after imports or after a method is finished 
  - lines that are **related** should be written **densely together**
- **Horizontal** openness
  - to accentuate operators / operator precedence
  - to separate parameters
  - use spaces to emphasize elements and indent to make scopes visible

### Don’t repeat yourself (DRY) 

> Do not duplicate pieces of code!

- Copy & paste decreases...
  - **Maintainability**: Losing track of copies
  - **Understandability**
    - Code is less compact
    - An identical concept needs to be understood multiple times
  - **Evolvability**
    - Need to find and modify all copies, When removing bugs or changing behaviour
- Duplicated code fosters errors and inconsistencies

### Keep it simple, stupid (KISS) 

> “Make everything as **simple** as possible, but not simpler”
>
> — *Albert Einstein*

- Good code is easy to understand by anybody

- Good code addresses the problem adequately

- For example, if an `IEnumerable` is suitable, do not use an`ICollection` or even an `IList`

- Techniques which help ensure that your code is understandable by others: 

  - Code reviews

  - Pair programming

### You ain’t gonna need it (YAGNI) 

> Only implement required features!

- Featurism is **costly**:

  - unrequested features need to be tested, documented

  - over-engineered systems sacrifice maintainability, as they are overly complex (KISS)

- Beware of **optimisations**!

  - Often merely treat symptoms
  - Too costly to be done prematurely

### Single Level of Abstraction (SLA)

- Newspaper metaphor:

  - Good newspaper articles are well-ordered

  - Navigation with details increasing: 
    - headline (very high abstraction)
    - text with synopsis (high abstraction)
    - rest (details)

- **Statements within a function** should be at the same abstraction level 

  - if not, extract expressions/statements of higher detail into an own method

- **Functions in a class**: The abstraction level should decrease depth- first when reading from top to bottom

## Refactoring

> *If it stinks, change it.*

- Methods tend to grow during development

- Bad odour (smell) of a long method arises

- What to do? Extract cohesive parts into new methods

  ![截屏2020-11-15 16.15.51](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2016.15.51.png)

### What is Refactoring?

> A „**disciplined technique for restructuring** an existing body of code, altering its internal structure **without changing its external behavior**.“ 
>
> *— M. Fowler*

### **The First Rule in Refactoring**

**Refactor with tests only!**

- Good tests help to prevent introducing bugs into the program through refactoring

### Bad Smells

Bad code smells: symptoms for deeper problems

- **Long method**: having code blocks lead by comments 
  - 👨‍⚕️Cure: Extract Method: extract commented block
- **Duplicated code**
- **Feature envy**: class A excessively calls another class B’s methods 
  - 👨‍⚕️Cure: parts of A’s methods want to be in class B
    1. Extract Method: extract code block calling class B
    2. Move Method: move extracted part to class B
- **Data class**: class merely holds data ("dumb data holder")
  - 👨‍⚕️Cure: enforce information hiding principle, collect functionality
    1. Encapsulate field: getter/setter instead of public access 
    2. Remove setting method: only for read-only values
    3. Move method: collect functionality implemented elsewhere 
       - think about responsibilities of the class

- **Large/God class**: class tries to do too much
- **Inappropriate intimacy**: class has dependencies on implementation details of another class
- ...

> More catelog see: https://www.refactoring.com/catalog/index.html

### **When to Refactor?**

- It is not that simple to find out **when to refactor**

- So-called “**bad smells**” in code may give a good indication when refactoring is worthwhile

- More general guidelines
  - when you find yourself looking up details frequently
    - *what was the order of the method parameters again?*
    - *where was this method again and what does it do?*
  - when you feel the need to write a **comment**
    - **try to refactor the code so that the comment becomes superfluous**

### Limitations

- May influence performance negatively 🤪
  - However, it is recommended to do the refactoring first
  - and the performance tuning on the cleaner code afterwards

## Appendix

### Separation of Concerns (SoC)

> Each module should be focused on a single concern.

- 👍 Benefits
  - Loose coupling, high cohesion
  - Better testability: each test stays focused on one module
- Some concerns may crosscut a system‘s core concerns
  - Typical crosscutting concerns: 
    - Tracing/Logging
    - Security 
    - Transactionality 
    - Caching
  - Aspect Oriented Programming (AOP) provides adequate concepts

### Order of Implementation

For the implementation (and unit testing later) always try to **move from the least-coupled to the most-coupled classes**

- avoids unnecessary creation of “stubs”

Example

![截屏2020-11-15 16.47.49](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-15%2016.47.49.png)

### Use a Version Control System

- **History** of commented changes

- Shared working in a team, even on same artefacts Branching and merging

- **Tagging** versions as pre-release etc.

- **Reverting** to previous revisions 

  - reduces fears of breaking code

  - encourages a programmer‘s willingness to refactor code

### Test First

- Test-Driven Development

- Clean tests should follow the **F.I.R.S.T.** rules

  - **F**ast: to run them frequently
  - **I**ndependent: A failing test does not influence others
  - **R**epeatable: in any environment, so there is no excuse for failing tests

  - **S**elf-Validating: Tests either pass or fail automatically
  - **T**imely: Tests are written right before production code

- Tests should follow same standards as production code

  - and be executed in a continuous manner 
    - so-called continuous integration
    - reduces fear of breaking code





  