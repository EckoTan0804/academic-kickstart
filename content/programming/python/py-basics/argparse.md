---
# Title, summary, and position in the list
linktitle: "argparse"
summary: ""
weight: 20

# Basic metadata
title: "argparse: Command line arguments parsing"
date: 2020-11-19
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
        weight: 10
---

## Basic usage

```python
import argparse

def get_args_parser():
    # set up a parser
    parser = argparse.ArgumentParser(description=f"{program_description}")
    
    # define parsing rules and operations for different arguments 
    # parser.add_argument()
    # ...
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    
    # parse command line 
    args = parser.parse_args()
    
    # further operations based on parsed arguments
    
```

## `add_argument()`

Parameters:

- [name or flags](https://docs.python.org/3/library/argparse.html#name-or-flags) - Either a name or a list of option strings, e.g. `foo` or `-f, --foo`.
- [action](https://docs.python.org/3/library/argparse.html#action) - The basic type of action to be taken when this argument is encountered at the command line.
- [nargs](https://docs.python.org/3/library/argparse.html#nargs) - The number of command-line arguments that should be consumed.
- [const](https://docs.python.org/3/library/argparse.html#const) - A constant value required by some [action](https://docs.python.org/3/library/argparse.html#action) and [nargs](https://docs.python.org/3/library/argparse.html#nargs) selections.
- [default](https://docs.python.org/3/library/argparse.html#default) - The value produced if the argument is absent from the command line.
- [type](https://docs.python.org/3/library/argparse.html#type) - The type to which the command-line argument should be converted.
- [choices](https://docs.python.org/3/library/argparse.html#choices) - A container of the allowable values for the argument.
- [required](https://docs.python.org/3/library/argparse.html#required) - Whether or not the command-line option may be omitted (optionals only).
- [help](https://docs.python.org/3/library/argparse.html#help) - A brief description of what the argument does.
- [metavar](https://docs.python.org/3/library/argparse.html#metavar) - A name for the argument in usage messages.
- [dest](https://docs.python.org/3/library/argparse.html#dest) - The name of the attribute to be added to the object returned by [`parse_args()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args).

### name or flags

The [`add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument) method must know whether an **optional** argument, or a **positional** argument is expected. Therefore, the first arguments passed to [`add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument) must  be either a series of flags, or a simple argument name.

**When [`parse_args()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args) is called, optional arguments will be identified by the `-` prefix, and the remaining arguments will be assumed to be positional.**

For example, an optional argument could be created like:

```python
parser.add_argument('-f', '--foo')
```

A positional argument could be created like:

```python
parser.add_argument('bar')
```

### action

- The `action` keyword argument specifies how the command-line arguments should be 

- handled. The supplied actions are:

  - `'store'` 

    This just stores the argument’s value.

  - `'store_const'` 

    This stores the value specified by the [const](https://docs.python.org/3/library/argparse.html#const) keyword argument. The `'store_const'` action is most commonly used with optional arguments that specify some sort of flag

  - `'store_true'` and `'store_false'` 

    These are special cases of `'store_const'` used for storing the values `True` and `False` respectively. In addition, they create default values of `False` and `True` respectively.

  - `'append'` 

    This stores a list, and appends each argument value to the list. This is useful to allow an option to be specified multiple times.

  - `'append_const'` 

    This stores a list, and appends the value specified by the [const](https://docs.python.org/3/library/argparse.html#const) keyword argument to the list.

  - `'count'` 

    This counts the number of times a keyword argument occurs.

  - `'help'` 

    This prints a complete help message for all the options in the current parser and then exits. By default a help action is automatically added to the parser.

  - `'version'` 

    This expects a `version=` keyword argument in the [`add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument) call, and prints version information and exits when invoked

  - `'extend'` 

    This stores a list, and extends each argument value to the list.

### nargs

The `nargs` specifies the number of command-line arguments that should be consumed.

| Supported value | Meaning                                       |
| --------------- | --------------------------------------------- |
| `N`             | The absolute number of arguments (e.g., `3`). |
| `?`             | 0 or 1 argument                               |
| `*`             | 0 or all arguments                            |
| `+`             | All, and at least one, argument               |

### const

The `const` argument of [`add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument) is used to hold constant values that are not read from the command line but are required for the various [`ArgumentParser`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser) actions. 

### default

All optional arguments and some positional arguments may be omitted at the command line. The `default` keyword argument of [`add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument) specifies what value should be used if the command-line argument is not present.

### type

By default, [`ArgumentParser`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser) objects read command-line arguments in as simple strings. The `type` keyword argument of [`add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument) allows any necessary type-checking and type conversions to be performed. 

### choices

- Some command-line arguments should be selected from a restricted set of values. 
- These can be handled by passing a container object as the *choices* keyword argument to [`add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument). 
- When the command line is parsed, argument values will be checked, and an error message will be displayed if the argument was not one of the acceptable values

### required

An argument is made required with the `required` option. If the required argument is not given, an error will be raised.

### help

- The `help` value is a string containing a brief description of the argument. 
- When a user requests help (usually by using `-h` or `--help` at the command line), these `help` descriptions will be displayed with each argument:

### metavar

- When [`ArgumentParser`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser) generates help messages, it needs some way to refer to each expected argument.
- By default
  - ArgumentParser objects use the [dest](https://docs.python.org/3/library/argparse.html#dest) value as the “name” of each object.
  - for positional argument actions, the [dest](https://docs.python.org/3/library/argparse.html#dest) value is used directly
    - E.g. a single positional argument with `dest='bar'` will be referred to as `bar`
  - for optional argument actions, the [dest](https://docs.python.org/3/library/argparse.html#dest) value is uppercased
    - E.g. A single optional argument `--foo` that should be followed by a single command-line argument will be referred to as `FOO`.

The `metavar` option gives the argument a name to the expected value displayed in error and help outputs.

### dest

- Most [`ArgumentParser`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser) actions add some value as an attribute of the object returned by [`parse_args()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args). The name of this attribute is determined by the `dest` keyword argument of [`add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument).

  - For positional argument actions, `dest` is normally supplied as the first argument to [`add_argument()`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument)	

    E.g. 

    ```python
    parser.add_argument('bar')
    ```

    The value of `dest` is `bar`

  - For optional argument actions, the value of `dest` is normally inferred from the option strings.

    - [`ArgumentParser`](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser) generates the value of `dest` by taking the first long option string and stripping away the initial `--` string.
    - If no long option strings were supplied, `dest` will be derived from the first short option string by stripping the initial `-` character.
    - Any internal `-` characters will be converted to `_` characters to make sure the string is a valid attribute name.

    E.g. 

    ```python
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of the person")
    ```

    The value of `dest` is `name`. After calling 

    ```python
    args = parser.parse_args()
    ```

    we can use the argument with `args.name`



## Example

Assume we have a python file ***person-info.py***:

```python
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="Person info program")
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of the person")
    parser.add_argument("-a", "--age", type=int, required=True, help="Age of the person")
    parser.add_argument(
        "-g",
        metavar="gender",
        dest="gender",
        default="male",
        choices=["male, female"],
        help="Gender of the person. (male, female)",
    )
    parser.add_argument("-s", "--single", dest="is_single", action="store_true", help="Is the person single?")
    parser.add_argument("-i", "--interests", metavar="interests", action="append", help="Interests of the person")
    parser.add_argument(
        "-e", metavar="education", dest="education", nargs="+", help="Educational experience of the person"
    )

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    print(f"Name: {args.name}")
    print(f"Age: {args.age}")
    print(f"Gender: {args.gender}")
    print(f"Is single?: {args.is_single}")
    print(f"Interests: {args.interests}")
    print(f"Education: {args.education}")

```

- Print the help information:

  ```bash
  $ python person-info.py -h
  ```

  ```
  usage: person-info.py [-h] -n NAME -a AGE [-g gender] [-s] [-i interests]
                        [-e education [education ...]]
  
  Person info program
  
  optional arguments:
    -h, --help            show this help message and exit
    -n NAME, --name NAME  Name of the person
    -a AGE, --age AGE     Age of the person
    -g gender             Gender of the person. (male, female)
    -s, --single          Is the person single?
    -i interests, --interests interests
                          Interests of the person
    -e education [education ...]
                          Educational experience of the person
  ```

- Run the script by given necesary arguments:

  ```bash
  $ python person-info.py -n Ecko -a 26 -s -i coding -e KIT KIT -i basketball -i fitness
  ```

  ```
  Name: Ecko
  Age: 26
  Gender: male
  Is single?: True
  Interests: ['coding', 'basketball', 'fitness']
  Education: ['KIT', 'KIT']
  ```

- Violation by not given age arguement `-a` :

  ```bash
  python person-info.py -n Ecko  -s -i coding -e KIT KIT -i basketball -i fitness 
  ```

  ```
  usage: person-info.py [-h] -n NAME -a AGE [-g gender] [-s] [-i interests]
                        [-e education [education ...]]
  person-info.py: error: the following arguments are required: -a/--age
  ```

## Reference

- Python documentation: [`argparse`](https://docs.python.org/3/library/argparse.html#module-argparse) — Parser for command-line options, arguments and sub-commands[¶](https://docs.python.org/3/library/argparse.html#module-argparse)

- [Python argparse tutorial](http://zetcode.com/python/argparse/)
- [argparse简要用法总结](https://vra.github.io/2017/12/02/argparse-usage/)
- [argparse – Command line option and argument parsing](https://pymotw.com/2/argparse/)