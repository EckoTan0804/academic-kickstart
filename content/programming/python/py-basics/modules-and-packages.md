---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 17

# Basic metadata
title: "Modules and Packages"
date: 2020-10-29
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics"]
categories: ["Coding"]
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
    python:
        parent: py-basics
        weight: 7
---

**Modular programming** refers to *the process of breaking a large, unwieldy programming task into separate, smaller, more manageable subtasks or **modules***. Individual modules can then be cobbled together like building blocks to create a larger application.

Advantages to **modularizing** code in a large application:

- **Simplicity**

  Rather than focusing on the entire problem at hand, a module typically focuses on one relatively small portion of the problem. If you’re working on a single module, you’ll have a smaller problem domain to wrap your head around. This makes development easier and less error-prone.

- **Maintainability** 

  Modules are typically designed so that they enforce logical boundaries between different problem domains. If modules are written in a way that minimizes interdependency, there is decreased likelihood that modifications to a single module will have an impact on other parts of the program. (You may even be able to make changes to a module without having any knowledge of the application outside that module.) This makes it more viable for a team of many programmers to work collaboratively on a large application.

- **Reusability** 

  Functionality defined in a single module can be easily reused (through an appropriately defined interface) by other parts of the application. This eliminates the need to duplicate code.

- **Scoping**

  Modules typically define a separate [**namespace**](https://realpython.com/python-namespaces-scope/), which helps avoid collisions between identifiers in different areas of a program. (One of the tenets in the [Zen of Python](https://www.python.org/dev/peps/pep-0020) is *Namespaces are one honking great idea—let’s do more of those!*)

**Functions**, **modules** and **packages** are all constructs in Python that promote code modularization.



## Python modules

- Different ways to define a **module** in Python:
  - **A module can be written in Python itself**.
  - A module can be written in **C** and loaded dynamically at run-time, like the `re` ([**regular expression**](https://realpython.com/regex-python/)) module.
  - A **built-in** module is intrinsically contained in the interpreter, like the [`itertools` module](https://realpython.com/python-itertools/).

- A module’s contents are accessed the same way in all three cases: with the `import`statement.
- Build a module written in python:
  - Create a file that contains legitimate Python code
  - Give the file a name with a `.py` extension.

- Example: 

  *mod.py*

  ```python
  s = "If Comrade Napoleon says it, it must be right."
  a = [100, 200, 300]
  
  def foo(arg):
      print(f'arg={arg}')
  
  class Foo:
      pass
  ```

  Several objects are defined in `mod.py`:

  - `s` (a string)
  - `a` (a list)
  - `foo()` (a function)
  - `Foo` (a class)

  These objects can be accessed by **importing** the module as follows (assuming `mod.py` is in an appropriate location):

  ```python
  >>> import mod
  >>> print(mod.s)
  If Comrade Napoleon says it, it must be right.
  >>> mod.a
  [100, 200, 300]
  >>> mod.foo(['quux', 'corge', 'grault'])
  arg = ['quux', 'corge', 'grault']
  >>> x = mod.Foo()
  >>> x
  <mod.Foo object at 0x03C181F0>
  ```

## The Module Search Path

When the interpreter executes the

```python
import mod
```

statement, it searches for `mod.py` in a list of directories assembled from the following sources:

- The directory from which the input script was run or the **current directory** if the interpreter is being run interactively
- The list of directories contained in the [`PYTHONPATH`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH) environment variable, if it is set. (The format for `PYTHONPATH` is OS-dependent but should mimic the `PATH`environment variable.)
- An installation-dependent list of directories configured at the time Python is installed

The resulting search path is accessible in the Python variable `sys.path`, which is obtained from a module named `sys`.

Thus, to ensure your module is found, you need to do one of the following:

- Put `mod.py` in the directory where the input script is located or the **current directory**, if interactive
- Modify the `PYTHONPATH` environment variable to contain the directory where `mod.py`is located before starting the interpreter
  - **Or:** Put `mod.py` in one of the directories already contained in the `PYTHONPATH` variable
- Put `mod.py` in one of the installation-dependent directories, which you may or may not have write-access to, depending on the OS

One additional option: you can put the module file in any directory of your choice and then modify `sys.path` at run-time so that it contains that directory.

## The `import` statement

### `import <module_name>`

Note that this *does not* make the module contents *directly* accessible to the caller.

- Each module has its own **private symbol table**, which serves as the global symbol table for all objects defined *in the module*. Thus, a module creates a separate **namespace**
- The statement `import <module_name>` only places `<module_name>` in the caller’s symbol table. The *objects* that are defined in the module *remain in the module’s private symbol table*.
- From the caller, objects in the module are only accessible when prefixed with `<module_name>` via **dot notation**

In our example, after the following `import` statement, `mod` is placed into the local symbol table. But `s` and `foo` remain in the module’s private symbol table and are not meaningful in the local context:

```python
>>> s
NameError: name 's' is not defined
>>> foo('quux')
NameError: name 'foo' is not defined
```

To be accessed in the local context, names of objects defined in the module must be **prefixed** by `mod`:

```python
>>> mod.s
'If Comrade Napoleon says it, it must be right.'
>>> mod.foo('quux')
arg = quux
```

### `from <module_name> import <name(s)>`

Because this form of import places the object names directly into the caller’s symbol table, any objects that already exist with the same name will be overwritten:

```python
>>> a = ['foo', 'bar', 'baz']
>>> a
['foo', 'bar', 'baz']

>>> from mod import a
>>> a
[100, 200, 300]
```

Indiscriminately `import` everything from a module at one fell swoop:

```python
from <module_name> import *
```

This will place the names of *all* objects from `<module_name>` into the local symbol table, with the exception of any that begin with the underscore (`_`) character.

{{% alert warning %}} 

This isn’t necessarily recommended in large-scale production code. It’s a bit dangerous because you are entering names into the local symbol table en masse. Unless you know them all well and can be confident there won’t be a conflict, you have a decent chance of overwriting an existing name inadvertently.

{{% /alert %}}

### `from <module_name> import <name> as <alt_name>[, <name> as <alt_name> …]`

This makes it possible to place names directly into the local symbol table but avoid conflicts with previously existing names

```python
>>> s = 'foo'
>>> a = ['foo', 'bar', 'baz']

>>> from mod import s as string, a as alist
>>> s
'foo'
>>> string
'If Comrade Napoleon says it, it must be right.'
>>> a
['foo', 'bar', 'baz']
>>> alist
[100, 200, 300]
```

### `import <module_name> as <alt_name>`

Import an entire module under an alternate name

```python
>>> import mod as my_module
>>> my_module.a
[100, 200, 300]
>>> my_module.foo('qux')
arg = qux
```

## The `dir()` function

The built-in function `dir()` returns a list of defined names in a namespace. Without arguments, it produces an alphabetically sorted list of names in the current **local symbol table**

## Executing a Module as a Script

Distinguish between when the file is loaded as a module and when it is run as a standalone script: 

- When a `.py` file is imported as a module, Python sets the special **dunder** variable `__name__` to the name of the module. 
- If a file is run as a standalone script, `__name__` is (creatively) set to the string `'__main__'`. 

Using this fact, we can discern which is the case at run-time and alter behavior accordingly:

```python
s = "If Comrade Napoleon says it, it must be right."
a = [100, 200, 300]

def foo(arg):
    print(f'arg = {arg}')

class Foo:
    pass

if (__name__ == '__main__'):
    print('Executing as standalone script')
    print(s)
    print(a)
    foo('quux')
    x = Foo()
    print(x)
```

Now, if we run as a script, we get output:

```python
>>> python mod.py
Executing as standalone script
If Comrade Napoleon says it, it must be right.
[100, 200, 300]
arg = quux
<__main__.Foo object at 0x03450690>
```

But if you import as a module, you don’t:

```python
>>> import mod
>>> mod.foo('grault')
arg = grault
```

Modules are often designed with the capability to run as a standalone script for purposes of testing the functionality that is contained within the module. This is referred to as **[unit testing](https://realpython.com/python-testing/).** 

## Reloading a module

For reasons of efficiency, a module is only loaded once per interpreter session. A module can contain executable statements as well, usually for initialization. Be aware that these statements will only be executed the first time a module is imported. If you make a change to a module and need to reload it, you need to either restart the interpreter or use a function called `reload()` from module `importlib`.

## Python packages

**Packages** allow for a hierarchical structuring of the module namespace using **dot notation**. In the same way that **modules** help avoid collisions between global variable names, **packages** help avoid collisions between module names.

Creating a **package** is quite straightforward, since it makes use of the operating system’s inherent hierarchical file structure. Consider the following arrangement:

![Image](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/pkg1.9af1c7aea48f.png)

Here, there is a directory named `pkg` that contains two modules, `mod1.py` and `mod2.py`. The contents of the modules are:

- *mod1.py*

  ```python
  def foo():
      print('[mod1] foo()')
  
  class Foo:
      pass
  ```

- *mod2.py*

  ```python
  def bar():
      print('[mod2] bar()')
  
  class Bar:
      pass
  ```

Given this structure, if the `pkg` directory resides in a location where it can be found (in one of the directories contained in `sys.path`), we can refer to the two **modules** with **dot notation** (`pkg.mod1`, `pkg.mod2`) and import them with the syntax we are already familiar with

- `import <module_name>[, <module_name> ...]`

  ```python
  >>> import pkg.mod1, pkg.mod2
  >>> pkg.mod1.foo()
  [mod1] foo()
  >>> x = pkg.mod2.Bar()
  >>> x
  <pkg.mod2.Bar object at 0x033F7290>
  ```

- `from <module_name> import <name(s)>`

  ```python
  >>> from pkg.mod1 import foo
  >>> foo()
  [mod1] foo()
  ```

- `from <module_name> import <name> as <alt_name>`

  ```python
  >>> from pkg.mod2 import Bar as Qux
  >>> x = Qux()
  >>> x
  <pkg.mod2.Bar object at 0x036DFFD0>
  ```

We can import modules with these statements as well:

```python
from <package_name> import <modules_name>[, <module_name> ...]
from <package_name> import <module_name> as <alt_name>
```

{{% alert note %}} 

Importing the package doesn't do much of anything useful. In particular, it does not place any of the modules in pkg into the local namespace

{{% /alert %}}

## Package Initialization

If a file named `__init__.py` is present in a package directory, it is invoked when the package or a module in the package is imported. This can be used for execution of **package initialization code**, such as initialization of package-level data.

![Image](https://files.realpython.com/media/pkg2.dab97c2f9c58.png)

Consider the following `__init__.py` file:

`__init__.py`:

```python
print(f'Invoking __init__.py for {__name__}')
A = ['quux', 'corge', 'grault']
```

Now when the package is imported, the global list `A` is initialized.

`__init__.py` can also be used to effect automatic importing of modules from a package. If `__init__.py` in the `pkg` directory contains the following:

```python
print(f'Invoking __init__.py for {__name__}')
import pkg.mod1, pkg.mod2
```

then when we execute `import pkg`, modules `mod1` and `mod2` are imported automatically:

```python
>>> import pkg
Invoking __init__.py for pkg
>>> pkg.mod1.foo()
[mod1] foo()
>>> pkg.mod2.bar()
[mod2] bar()
```

{{% alert note %}} 

Note:

- Before **Python 3.3**, an `__init__.py` file **must** be present in the package directory when creating a package. It used to be that the very presence of `__init__.py` signified to Python that a package was being defined. The file could contain initialization code or even be empty, but it **had** to be present.
- Starting with **Python 3.3**, [Implicit Namespace Packages](https://www.python.org/dev/peps/pep-0420) were introduced. These allow for the creation of a package without any `__init__.py` file. Of course, it **can** still be present if package initialization is needed. But it is no longer required.

{{% /alert %}}

## Importing `*` from a package

```python
from <package_name> import *
```

Python follows this convention:

If the `__init__.py` file in the **package** directory contains a **list** named `__all__`, it is taken to be a list of modules that should be imported when the statement `from <package_name> import *` is encountered.

For example, consider the following structure:

![Illustration of hierarchical file structure of Python packages](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/pkg3.d2160908ae77.png)

Suppose we create an `__init__.py` in the `pkg` directory like this:

***`pkg/__init__.py`***

```python
__all__ = [
        'mod1',
        'mod2',
        'mod3',
        'mod4'
        ]
```

Now `from pkg import *` imports all four modules:

```python
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__']

>>> from pkg import *
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__', 'mod1', 'mod2', 'mod3', 'mod4']
>>> mod2.bar()
[mod2] bar()
>>> mod4.Qux
<class 'pkg.mod4.Qux'>
```



In summary, `__all__` is used by both **packages** and **modules** to control what is imported when `import *` is specified. But *the default behavior differs*:

- For a package, when `__all__` is not defined, `import *` does not import anything.
- For a module, when `__all__` is not defined, `import *` imports everything (except—you guessed it—names starting with an underscore).



## Subpckages

Packages can contain nested subpackages to arbitrary depth. For example:

![Image](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/pkg4.a830d6e144bf.png)

Importing still works the same as shown previously. Syntax is similar, but additional dot notation is used to separate package name from subpackage name:

```python
import <package_name>.<subpackage_name>.<module_name>
```













## Reference

[Python Modules and Packages – An Introduction](https://realpython.com/python-modules-packages/#importing-from-a-package)