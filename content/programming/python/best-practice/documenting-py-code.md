---
# Basic info
title: "Documenting Python Code"
date: 2020-07-06
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Best practice"]
categories: ["Coding"]
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
    python:
        parent: py-best-practice
        weight: 2

weight: 22
---

Source: [Documenting Python Code: A Complete Guide](https://realpython.com/documenting-python-code/)

## Commenting vs. Documenting Code

|                 | Description                   | Audience                   |
| --------------- | ----------------------------- | -------------------------- |
| **Commenting**  | Purpose and design of code    | Maintainers and developers |
| **Documenting** | Use and functionality of code | Users                      |

## Commenting Code

Comments are created in Python using the pound sign (`#`) and should be **brief statements no longer than a few sentences**.

~~~python
def hello_world():
  # A simple comment preceding a simple print statement
  print("Hello World!")
~~~

According to [PEP 8](http://pep8.org/#maximum-line-length), comments should have a **maximum length of 72** characters. This is true even if your project changes the max line length to be greater than the recommended 80 characters. 

~~~python
def hello_long_world():
    # A very long statement that just goes on and on and on and on and
    # never ends until after it's reached the 80 char limit
    print("Hellooooooooooooooooooooooooooooooooooooooooooooooooooooooo World")
~~~

### Purpose of Commenting Code

- **Planning and reviewing**
  When developing new portion of code, first use comments as a way of planning or outlineing that section of code. **Remember to remove these comments once the actual coding has been implemented and reviewed/tested**

  ~~~python
  def new_function():
    # Step 1
    # Step 2
    # Step 3
    
  ~~~

- **Code description** 
  Explain the intent of specific sections of code

- **Algorithmic description**
  When algorithms are used, especially complicated ones, it can be useful to explain how the algorithm works or how it’s implemented within your code.

  ~~~python
  # Using quick sort for performance gains
  ~~~

- **Tagging:**
  The use of tagging can be used to label specific sections of code where known issues or areas of improvement are located. 
  Some examples are: `BUG`, `FIXME`, and `TODO`.

  ~~~python
  # TODO: Add condition for when val is None### 
  ~~~

### Rules of Commenting

- Should be kept brief and focused
- Avoid using long comments when possible

Essential rules as [suggested by Jeff Atwood](https://blog.codinghorror.com/when-good-comments-go-bad/):

- Keep comments as close to the code being described as possible.
- Don’t use complex formatting (such as tables or ASCII figures).
- Don’t include redundant information. Assume the reader of the code has a basic understanding of programming principles and language syntax.
- Design your code to comment itself. 💪

### Commenting Code via Type Hinting (Python 3.5+)

Type hinting was added to Python 3.5 and is an additional form to help the readers of your code. 

Example

~~~python
def hello_name(name: str) -> str:
  return ("Hello {name}")
~~~

You can immediately tell that 

- the function expects the input `name` to be of a type `str`, or string. 
- the expected output of the function will be of a type `str`, or string, as well.



## Documenting Code Base using Docstrings

### Docstings Background

**Docstrings are built-in strings that, when configured correctly, can help your users and yourself with your project’s documentation.**

Python also has the built-in function `help()` that prints out the objects docstring to the console. 

Example:

~~~python
def say_hello(name):
  """A simple function that says hello"""
  print(f"Hello {name})
~~~

~~~python
>>> help(say_hello)
Help on function say_hello in module __main__:

say_hello(name)
    A simple function that says hello
~~~

### Docstring Types

Docstring conventions:

- Are described within [PEP 257](https://www.python.org/dev/peps/pep-0257/)
- Purpose: provide your users with a brief overview of the object. 
- Should be kept concise enough to be easy to maintain but still be elaborate enough for new users to understand their purpose and how to use the documented object.

**In all cases, the docstrings should use the triple-double quote (`"""`) string format.** This should be done whether the docstring is multi-lined or not. 

At a bare minimum, a docstring should be a quick summary of whatever is it you’re describing and should be contained within a single line:

~~~python
"""This is a quick summary line used as a description of the object."""

~~~

Multi-lined docstrings are used to further elaborate on the object beyond the summary. All multi-lined docstrings have the following parts:

- A one-line summary line
- A blank line proceeding the summary
- Any further elaboration for the docstring
- Another blank line

~~~python
"""This is the summary line

This is the further elaboration of the docstring. Within this section,
you can elaborate further on details as appropriate for the situation.
Notice that the summary and the elaboration is separated by a blank new
line.
"""

# Notice the blank line above. Code should continue on this line.
~~~

All docstrings should have the same **max character length as comments (72 characters).** 

Three major categories:

- **Class Docstrings:** Class and class methods
- **Package and Module Docstrings:** Package, modules, and functions
- **Script Docstrings:** Script and functions

#### Class Docstrings

~~~python
class SimpleClass:
    """Class docstrings go here."""

    def say_hello(self, name: str):
        """Class method docstrings go here."""

        print(f'Hello {name}')
~~~

Class docstrings should contain the following information:

- A brief summary of its purpose and behavior
- Any public methods, along with a brief description
- Any class properties (attributes)
- Anything related to the interface for subclassers, if the class is intended to be subclassed

The class constructor parameters should be documented within the `__init__` class method docstring. 

Individual methods should be documented using their individual docstrings. Class method docstrings should contain the following:

- A brief description of what the method is and what it’s used for
- Any arguments (both required and optional) that are passed including keyword arguments
- Label any arguments that are considered optional or have a default value
- Any side effects that occur when executing the method
- Any exceptions that are raised
- Any restrictions on when the method can be called

Example:

~~~python
class Animal:
    """
    A class used to represent an Animal

    ...

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    says_str = "A {name} says {sound}"

    def __init__(self, name, sound, num_legs=4):
        """
        Parameters
        ----------
        name : str
            The name of the animal
        sound : str
            The sound the animal makes
        num_legs : int, optional
            The number of legs the animal (default is 4)
        """

        self.name = name
        self.sound = sound
        self.num_legs = num_legs

    def says(self, sound=None):
        """Prints what the animals name is and what sound it makes.

        If the argument `sound` isn't passed in, the default Animal
        sound is used.

        Parameters
        ----------
        sound : str, optional
            The sound the animal makes (default is None)

        Raises
        ------
        NotImplementedError
            If no sound is set for the animal or passed in as a
            parameter.
        """

        if self.sound is None and sound is None:
            raise NotImplementedError("Silent Animals are not supported!")

        out_sound = self.sound if sound is None else sound
        print(self.says_str.format(name=self.name, sound=out_sound))
~~~

#### Package and Module Docstrings

Package Docstrings:

- Should be placed at the top of the package’s `__init__.py` file
- Should list the modules and sub-packages that are exported by the package

Module Docstrings:

- Placed at the top of the file even before any imports
- Should include:
  - A brief description of the module and its purpose
  - A list of any classes, exception, functions, and any other objects exported by the module
  - Docstring for a module function should include the same items as a class method

#### Script Docstrings

Scripts: single file executables run from the console. 

Docstrings for scripts:

- Placed at the **top of the file** 
- should be documented well enough for users to be able to have a sufficient understanding of how to use the script
- Should be usable for its “usage” message, when the user incorrectly passes in a parameter or uses the `-h` option

Any custom or third-party imports should be listed within the docstrings to allow users to know which packages may be required for running the script

Example:

~~~python
"""Spreadsheet Column Printer

This script allows the user to print to the console all columns in the
spreadsheet. It is assumed that the first row of the spreadsheet is the
location of the columns.

This tool accepts comma separated value files (.csv) as well as excel
(.xls, .xlsx) files.

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * get_spreadsheet_cols - returns the column headers of the file
    * main - the main function of the script
"""

import argparse

import pandas as pd


def get_spreadsheet_cols(file_loc, print_cols=False):
    """Gets and prints the spreadsheet's header columns

    Parameters
    ----------
    file_loc : str
        The file location of the spreadsheet
    print_cols : bool, optional
        A flag used to print the columns to the console (default is
        False)

    Returns
    -------
    list
        a list of strings used that are the header columns
    """

    file_data = pd.read_excel(file_loc)
    col_headers = list(file_data.columns.values)

    if print_cols:
        print("\n".join(col_headers))

    return col_headers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_file',
        type=str,
        help="The spreadsheet file to pring the columns of"
    )
    args = parser.parse_args()
    get_spreadsheet_cols(args.input_file, print_cols=True)


if __name__ == "__main__":
    main()
~~~



### Docstring Formats

Some of the most common formats are the following:

| Formatting Type                                              | Description                                                  | Supported by Sphynx | Formal Specification |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------- | -------------------- |
| [Google docstrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) | Google’s recommended form of documentation                   | Yes                 | No                   |
| [reStructured Text](http://docutils.sourceforge.net/rst.html) | Official Python documentation standard; Not beginner friendly but feature rich | Yes                 | Yes                  |
| [NumPy/SciPy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) | NumPy’s combination of reStructured and Google Docstrings    | Yes                 | Yes                  |
| [Epytext](http://epydoc.sourceforge.net/epytext.html)        | A Python adaptation of Epydoc; Great for Java developers     | Not officially      | Yes                  |

The selection of the docstring format is up to you, but you should stick with the same format throughout your document/project. The following are examples of each type to give you an idea of how each documentation format looks.
