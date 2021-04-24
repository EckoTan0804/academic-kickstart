---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 1

# Basic metadata
title: "SOLID Principles"
date: 2020-11-06
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Software Engineering", "Design Pattern"]
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
        parent: design-patterns
        weight: 1
---

## **S**ingle responsibility principle

The **single responsibility principle (SRP)** states that a software component (in general, a class) must have only ONE responsibility. 

This design principle helps us build more cohesive abstractions

- objects that do one thing, and just one thing
- Avoid: objects with multiple responsibilites (aka **god-objects**)
  - These objects group different (mostly unrelated) behaviors, thus making them harder to maintain.

Goal: **Classes are designed in such a way that most of their properties and their attributes are used by its methods, most of the time.**  When this happens, we know they are related concepts, and therefore it makes sense to group them under the same abstraction.

There is another way of looking at this principle. If, when looking at a class, we find methods that are *mutually exclusive* and do not relate to each other, they are the *different* responsibilities that have to be broken down into smaller classes.

### Example

In this example, we are going to create the case for an application that is in charge of reading information about events from a source (this could be log files, a database, or many more sources), and identifying the actions corresponding to each particular log.

A design that <span style="color:red">fails</span> to conform to the SRP would look like this:

![截屏2020-11-14 21.27.01](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-14%2021.27.01.png)

```python
class SystemMonitor:
  
  def load_activity(self):
    """Get the events from a source, to be processed."""
    
  def identify_events(self):
    """Parse the source raw data into events (domain objects)."""
    
  def stream_events(self):
    """Send the parsed events to an external agent."""
```

🔴 Problem:

- It defines an interface with a set of methods that correspond to actions that are orthogonal: each one can be done independently of the rest.
- This design flaw makes the class rigid, inflexible, and error-prone because it is hard to maintain. 
  - Consider the loader method (`load_activity`), which retrieves the information from a particular source. Regardless of how this is done (we can abstract the implementation details here), it is clear that it will have its own sequence of steps, for instance connecting to the data source, loading the data, parsing it into the expected format, and so on. If any of this changes (for example, we want to change the data structure used for holding the data), the SystemMonitor class will need to change. Ask yourself whether this makes sense. Does a system monitor object have to change because we changed the representation of the data? NO!
  - The same reasoning applies to the other two methods. If we change how we fingerprint events, or how we deliver them to another data source, we will end up making changes to the same class.

This class is rather fragile, and not very maintainable. There are lots of different reasons that will impact on changes in this class. Instead, we want external factors to impact our code **as little as possible**. The solution, again, is to create smaller and more cohesive abstractions.

#### **Solution: Distributing responsibilities**

To make the solution more maintainable, we separate every method into a different class. This way, each class will have a single responsibility:

![截屏2020-11-14 21.33.14](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-14%2021.33.14.png)

The same behavior is achieved by using an object that will interact with instances of these new classes, using those objects as collaborators, but the idea remains that each class encapsulates a specific set of methods that are independent of the rest. Now changing any of these classes do not impact the rest, and all of them have a clear and specific meaning.

👍 Advantaegs

- Changes are now local, the impact is minimal, and each class is easier to maintain.
- The new classes define interfaces that are not only more maintainable but also reusable.

## **O**pen/closed principle

The **open/closed principle** (**OCP**) states that a modele should be **open to extension but closed for modification.**

- we want our code to be extensible, to adapt to new requirements, or changes in the domain problem. 
- when something new appears on the domain problem, we only want to add new things to our model, not change anything existing that is closed to modification.

### Example of maintainability perils for NOT following the open/closed principle

In this example, we have a part of the system that is in charge of identifying events as they occur in another system, which is being monitored. At each point, we want this component to identify the type of event, correctly, according to the values of the data that was previously gathered.

First attempt:

```python
class Event:
  def __init__(self, raw_data):
    self.raw_data = raw_data
    

class UnknownEvent(Event):
	"""A type of event that cannot be identified from its data."""
  
class LoginEvent(Event):
	"""A event representing a user that has just entered the system."""
  
class LogoutEvent(Event):
	"""An event representing a user that has just left the system."""
  
class SystemMonitor:
  """Identify events that occurred in the system."""
	def __init__(self, event_data):
    self.event_data = event_data
    
  def identify_event(self):
    if self.event_data["before"]["session"] == 0 and self.event_data["after"]["session"] == 1: 
      return LoginEvent(self.event_data)
    elif self.event_data["before"]["session"] == 1 and self.event_data["after"]["session"] == 0:
      return LogoutEvent(self.event_data)
    
    return UnknownEvent(self.event_data)
```

🔴 Problems

- The logic for determining the types of events is centralized inside a monolithic method. As the number of events we want to support grows, this method will as well, and it could end up being a very long method. 🤪
- This method is not closed for modification. Every time we want to add a new type of event to the system, we will have to change something in this method 🤪

### Refactoring the events system for extensibility

In order to achieve a design that honors the open/closed principle, we have to design toward abstractions.

A possible alternative would be to think of this class as it collaborates with the events, and then we **delegate the logic for each particular type of event to its corresponding class**:

![截屏2020-11-14 21.46.43](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-14%2021.46.43.png)

Then we have to 

- add a new (polymorphic) method to each type of event with the single responsibility of determining if it corresponds to the data being passed or not, 
- and change the logic to go through all events, finding the right one.

```python
class Event:
    def __init__(self, raw_data):
        self.raw_data = raw_data
    
    @staticmethod
    def meets_condition(event_data: dict):
        return False

class UnknownEvent(Event):
    """A type of event that cannot be identified from its data"""


class LoginEvent(Event):

    @staticmethod
    def meets_condition(event_data: dict):
        return event_data["before"]["session"] == 0 and event_data["after"]["session"] == 1


class LogoutEvent(Event):
    
    @staticmethod
    def meets_condition(event_data: dict):
        return event_data["before"]["session"] == 1 and event_data["after"]["session"] == 0


class SystemMonitor:
    """Identify events that occurred in the system."""

    def __init__(self, event_data):
        self.event_data = event_data
    
    def identify_event(self):
        for event_cls in Event.__subclasses__():
            try:
                if event_cls.meets_condition(self.event_data):
                    return event_cls(self.event_data)
            except KeyError:
                continue
        return UnknownEvent(self.event_data)
```

👍 Advantages of this implementation:

- The `identify_event` method no longer works with specific types of event, but just with generic events that follow a common interface—they are all polymorphic with respect to the `meets_condition` method.

- Supporting new types of event is now just about creating a new class for that event that has to inherit from `Event` and implement its own `meets_condition()` method, according to its specific business logic.

  Imagine that a new requirement arises, and we have to also support events that correspond to transactions that the user executed on the monitored system. The class diagram for the design has to include such a new event type, as in the following:

  ![截屏2020-11-14 22.28.54](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-14%2022.28.54.png)

  We just need to add a `TransactionEvent` class like this:

  ```python
  class TransactionEvent(Event):
    	@staticmethod
      def meets_condition(event_data: dict):
        	return event_data["after"].get("transaction") is not None
  ```

  And we don't have to change anything else.:clap:

## **L**iskov's substitution principle

The main idea behind **Liskov's substitution principle** (**LSP**) is that for any class, a client should be able to use any of its subtypes indistinguishably, without even noticing, and therefore without compromising the expected behavior at runtime. This means that clients are completely isolated and unaware of changes in the class hierarchy.

More formally: if *S* is a subtype of *T*, then objects of type *T* may be replaced by objects of type *S*, without breaking the program.

This can be understood with the help of the following generic diagram:

![截屏2020-11-14 22.36.03](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-11-14%2022.36.03.png)

If the hierarchy is correctly implemented, the client class has to be able to work with instances of any of the subclasses without even noticing.







**TIP:** This training could take several hours depending on how many iterations you chose in the .cfg file. You will want to let this run as you sleep or go to work for the day, etc. However, Colab Cloud Service kicks you off it's VMs if you are idle for too long (30-90 mins).

To avoid this hold (CTRL + SHIFT + i) at the same time to open up the inspector view on your browser.

Paste the following code into your console window and hit **Enter**

```
function ClickConnect(){
console.log("Working"); 
document
  .querySelector('#top-toolbar > colab-connect-button')
  .shadowRoot.querySelector('#connect')
  .click() 
}
setInterval(ClickConnect,60000)
```











## **I**nterface segregation principle



## **D**ependency inversion principle