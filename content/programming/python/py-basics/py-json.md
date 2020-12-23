---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 21

# Basic metadata
title: "JSON"
date: 2020-12-09
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics", "JSON"]
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
        weight: 11
---

## TL;DR

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/py-json.png" title="Python and JSON conversion" numbered="true" >}}

JSON (**J**ava**S**cript **O**bject **N**otation) is a popular data format used for representing structured data. It's common to transmit and receive data between a server and web application in JSON format.

## JSON in Python 3

Python has a built-in package called `json`, which can be used to work with JSON data.

```python
import json
```



![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/5767796078149632.svg)





## JSON to Python

You can parse a **JSON string** using  `json.loads()` method. The method returns a dictionary.

For example:

```python
import json

person = '{"name": "Bob", "languages": ["English", "Fench"]}' # JSON string
person_dict = json.loads(person) # Python dictionary

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print(person_dict)

# Output: ['English', 'French']
print(person_dict['languages'])
```

To read a **file containing JSON object**, you can use `json.load()` method 

Suppose, you have a file named `person.json` which contains a JSON object.

```json
{
    "name": "Bob", 
    "languages": [
        "English", 
        "Fench"
    ]
}
```

Parse this file:

```python
import json

with open('person.json') as f:
  data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print(data)
```

### Json2Python Conversion table

| JSON          | Python Equivalent |
| ------------- | ----------------- |
| object        | dict              |
| array         | list              |
| string        | str               |
| number (int)  | int               |
| number (real) | float             |
| true          | True              |
| false         | False             |
| null          | None              |

## Python to JSON

You can convert a dictionary to **JSON string** using `json.dumps()` method.

```python
import json

person_dict = {
    'name': 'Bob',
    'age': 12,
    'children': None
}
person_json = json.dumps(person_dict)

# Output: {"name": "Bob", "age": 12, "children": null}
print(person_json)
```

To write JSON to a **file** in Python, we can use `json.dump()` method.

```python
import json

person_dict = {
    "name": "Bob",
    "languages": [
        "English", 
        "Fench"
    ],
	"married": True,
	"age": 32
}

with open('person.json', 'w') as json_file:
  json.dump(person_dict, json_file)
```

When you run the program, the `person.txt` file will be created. The file has following text inside it.

```
{"name": "Bob", "languages": ["English", "Fench"], "married": true, "age": 32}
```

### Python2Json Conversion table

| Python                                 | JSON Equivalent |
| :------------------------------------- | :-------------- |
| dict                                   | object          |
| list, tuple                            | array           |
| str                                    | string          |
| int, float, int- & float-derived Enums | number          |
| True                                   | true            |
| False                                  | false           |
| None                                   | null            |

## Formatting

### Indent

Use the `indent` parameter to define the numbers of indents. For example:

```python
json.dumps(person_dict, indent=4)
```

### Order the result

Sort keys in ascending order using `sort_keys=True`. E.g.:

```python
json.dumps(person_dict, indent = 4, sort_keys=True)
```

## Reference

- [Python JSON](https://www.programiz.com/python-programming/json)
- [Python JSON tutotials in w3schools](https://www.w3schools.com/python/python_json.asp)

- [Python3 JSON 数据解析](https://www.runoob.com/python3/python3-json.html)

