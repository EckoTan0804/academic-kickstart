---
# Basic info
title: "Beautiful Python Code with PEP 8"
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
        weight: 1

weight: 201
---

Source: [How to Write Beautiful Python Code With PEP 8](https://realpython.com/python-pep8/) 

## Naming Conventions

> “Explicit is better than implicit.”
— The Zen of Python

‼️ Note: Never use l (\ell), O (zero), or I (capital i) single letter names as these can be mistaken for 1 and 0, depending on typeface:

### Naming styles

![Beautiful%20Python%20Code%20with%20PEP%208%20ad6a9a89c353410c813a83ebda6504c3/Untitled.png](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Untitled.png)

### How to choose names

The best way to name your objects in Python is to use **descriptive** names to make it clear what the object represents.

**Always try to use the most concise but descriptive names possible.**

❌

```python
# Not recommended
>>> x = 'John Smith'
>>> y, z = x.split()
>>> print(z, y, sep=', ')
'Smith, John'

```

✅

```python
>>> # Recommended
>>> name = 'John Smith'
>>> first_name, last_name = name.split()
>>> print(last_name, first_name, sep=', ')
'Smith, John'

```

## Code Layout

> “Beautiful is better than ugly.”
— The Zen of Python

### Blank lines

**Surround top-level functions and classes with two blank lines**

Example:

```python
class MyFirstClass:
    pass

class MySecondClass:
    pass

def top_level_function():
    return None

```

**Surround method definitions inside classes with a single blank line.**

Example:

```python
def first_method(self):
    return None

def second_method(self):
    return None

```

**Use blank lines sparingly inside functions to show clear steps.**

Example:

```python
def calculate_variance(number_list):
    sum_list = 0
    for number in number_list:
        sum_list = sum_list + number
    mean = sum_list / len(number_list)

    sum_squares = 0
    for number in number_list:
        sum_squares = sum_squares + number**2
    mean_squares = sum_squares / len(number_list)

    return mean_squares - mean**2

```

### Maximum Line Length and Line Breaking

**Lines should be limited to 79 characters.**

Outlines ways to allow statements to run over several lines:

- **Assume line continuation if code is contained within parentheses, brackets, or braces:**

    ```python
    def function(arg_one, arg_two,
             arg_three, arg_four):
        return arg_one

    ```

- **Use backslashes to break lines if it is impossible to use implied continuation:**

    ```python
    from mypkg import example1, \\
        example2, example3

    ```

- **if you can use implied continuation, then you should do so.**
    - If line breaking needs to occur around binary operators, like + and *, it should occur **before** the operator

        ❌

        ```python
        # Not Recommended
        total = (first_variable +
                 second_variable -
                 third_variable)

        ```

        ✅

        ```python
        # Recommended
        total = (first_variable
                 + second_variable
                 - third_variable)

        ```

## Indentation

- Use **4** consecutive spaces to indicate indentation.
- Prefer **spaces over tabs**.

### Indentation Following Line Breaks

When you’re using line continuations to keep lines to under 79 characters, it is useful to use indentation to improve readability. It allows the reader to **distinguish between two lines of code and a single line of code that spans two lines.**

There are two styles of indentation you can use:

1. align the indented block with the opening delimiter:

    ```python
        def function(arg_one, arg_two,
                     arg_three, arg_four):
            return arg_one

    ```

    Sometimes only 4 spaces are needed to align with the opening delimiter. This will often occur in if statements that span multiple lines as the `if`, space, and opening bracket make up 4 characters. In this case, it can be difficult to determine where the nested code block inside the if statement begins:

    Example:

    ```python
        x = 5
        if (x > 3 and
            x < 10):
            print(x)

    ```

    In this case, PEP 8 provides two alternatives to help improve readability:

    - Add a comment after the final condition.

        ```python
        x = 5
        if (x > 3 and
            x < 10):
            # Both conditions satisfied
            print(x)

        ```

    - Add extra indentation on the line continuation:

        ```python
        x = 5
        if (x > 3 and
                x < 10):
            print(x)

        ```

2. **hanging indent**: You can use a hanging indent to visually represent a continuation of a line of code.

    Example:

    ```python
    var = function(
        arg_one, arg_two,
        arg_three, arg_four)

    ```

    ‼️ When you’re using a hanging indent, **there must not be any arguments on the first line**.
    The following example is **not** PEP 8 compliant:

    ```python
    var = function(arg_one, arg_two,
        arg_three, arg_four)

    ```

    - When using a hanging indent, **add extra indentation** to distinguish the continued line from code contained inside the function.

        ❌

        ```python
        # Not Recommended
        def function(
           arg_one, arg_two,
           arg_three, arg_four):
           return arg_one

        ```

        Instead, it’s better to use a double indent on the line continuation. This helps you to distinguish between function arguments and the function body, improving readability:

        ✅

        ```python
        # Recommended
        def function(
                arg_one, arg_two,
                arg_three, arg_four):
            return arg_one

        ```

### Where to Put the Closing Brace

Two options for the position of the closing brace in implied line continuations:

- Line up the closing brace with the first non-whitespace character of the previous line:

    ```python
    list_of_numbers = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
        ]

    ```

- Line up the closing brace with the first character of the line that starts the construct:

    ```python
    list_of_numbers = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    ]

    ```

**Consistency is key, try to stick to one of the above methods.**

## Comments and Documentations

See: [Documenting Python Code](quiver:///notes/CE487BE2-FAC4-4997-A723-244CFC727EA2)

## Whitespace in Expressions and Statements

> “Sparse is better than dense.”— The Zen of Python

### Whitespace Around Binary Operators

Surround the following binary operators with a single space on either side:

- Assignment operators (`=`, `+=`, `-=`, and so forth)
- Comparisons (`==`, `!=`, `>`, `<`. `>=`, `<=`) and (`is`, `is not`, `in`, `not in`)
- Booleans (`and`, `not`, `or`)

‼️ **Note**:

- When `=` is used to assign a default value to a function argument, do **NOT** surround it with spaces.

    ✅

    ```python
    # Recommended
    def function(default_parameter=5):
        # ...

    ```

    ❌

    ```python
    # Not recommended
    def function(default_parameter = 5):
        # ...

    ```

- If there's more than one operator in a statement, **only add whitespace around the operators with the lowest priority**. especially when performing mathematical manipulation.

    ✅

    ```python
    # Recommended
    y = x**2 + 5
    z = (x+y) * (x-y)

    ```

    ❌

    ```python
    # Not recommended
    y = x ** 2 + 5
    z = (x + y) * (x - y)

    ```

    - Apply this rule to `if` statements where there are multiple conditions:

        ✅

        ```python
        # Recommended
        if x>5 and x%2==0:
            print('x is larger than 5 and divisible by 2!')

        ```

        ❌

        ```python
        # Not recommended
        if x > 5 and x % 2 == 0:
            print('x is larger than 5 and divisible by 2!')

        ```

    - In slices, colons act as a binary operators. Therefore, the rules outlined in the previous section apply, and there should be the **same amount of whitespace** either side. The following examples of list slices are valid:

    ```python
    list[3:4]

    # Treat the colon as the operator with lowest priority
    list[x+1 : x+2]

    # In an extended slice, both colons must be
    # surrounded by the same amount of whitespace
    list[3:4:5]
    list[x+1 : x+2 : x+3]

    # The space is omitted if a slice parameter is omitted
    list[x+1 : x+2 :]

    ```

### Summary

Surround most operator with whitespace, except:

- in function arguments
- combining multiple operators in one statement

## Programming Recommendations

> “Simple is better than complex.”— The Zen of Python

🎯 Goal: readability and simplicity

### Don’t compare boolean values to `True` or `False` using the equivalence operator.

❌

```python
# Not recommended
my_bool = 6 > 5
if my_bool == True:
    return '6 is bigger than 5'

```

✅

```python
# Recommended
if my_bool:
    return '6 is bigger than 5'

```

### Use the fact that empty sequences are falsy in `if` statements.

In Python any empty list, string, or tuple is [falsy](https://docs.python.org/3/library/stdtypes.html#truth-value-testing).

❌

```python
# Not recommended
my_list = []
if not len(my_list):
    print('List is empty!')

```

✅

```python
# Recommended
my_list = []
if not my_list:
    print('List is empty!')

```

### Use `is not` rather than `not ... is` in `if` statements.

❌

```python
# Not recommended
if not x is None:
    return 'x exists!'

```

✅

```python
# Recommended
if x is not None:
    return 'x exists!'

```

### Don’t use `if x:` when you mean `if x is not None:`.

❌

```python
# Not recommended
if arg:
    # Do something with arg...

```

✅

```python
# Recommended
if arg is not None:
    # Do something with arg...

```

### Use `.startswith()` and `.endswith()` instead of slicing.

- prefix

    ❌

    ```python
    # Not recommended
    if word[:3] == 'cat':
        print('The word starts with "cat"')

    ```

    ✅

    ```python
    # Recommended
    if word.startswith('cat'):
        print('The word starts with "cat"')

    ```

- suffix

    ❌

    ```python
    # Not recommended
    if file_name[-3:] == 'jpg':
        print('The file is a JPEG')

    ```

    ✅

    ```python
    # Recommended
    if file_name.endswith('jpg'):
        print('The file is a JPEG')

    ```

## Tips and Tricks to Help Ensure Your Code Follows PEP 8

**Never ignore PEP 8!!!** 

### Linters

Linters are programs that analyze code and flag errors. They provide suggestions on how to fix the error.

Best linters for Python code:

- **[`pycodestyle`](https://pypi.org/project/pycodestyle/)** is a tool to check your Python code against some of the style conventions in PEP 8.

    Install `pycodestyle` using `pip`:

    ```python
    $ pip install pycodestyle
    ```

- **[`flake8`](https://pypi.org/project/flake8/)** is a tool that combines a debugger, `pyflakes`, with `pycodestyle`.

    Install `flake8` using `pip`:

    ```python
    $ pip install flake8
    ```

### Autoformatters

Autoformatters are programs that refactor your code to conform with PEP 8 automatically. Once such program is [`black`](https://pypi.org/project/black/), which autoformats code following *most* of the rules in PEP 8.

Install `black` using `pip`. It requires Python 3.6+ to run:

```bash
$ pip install black
```