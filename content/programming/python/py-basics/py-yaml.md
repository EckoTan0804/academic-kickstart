---
# Title, summary, and position in the list
linktitle: "YAML"
summary: ""
weight: 22

# Basic metadata
title: "YAML in Python"
date: 2020-12-09
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics", "YAML"]
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
        weight: 12
---

## YAML

- YAML = *YAML Ain't Markup Language*
- Human-readable data-serialization language
- It is commonly used for configuration files, but it is also used in data storage (e.g. debugging output) or transmission (e.g. document headers).
- Filename extension: `.yaml`

## Use YAML in Python

PyYAML is a YAML parser and emitter for Python.

```bash
$ pip install pyyaml
```

### Read YAML

Let's say we have a YAML file called `person_info.yaml`:

```yaml
name: "Ecko"
age: 20
hobby:
    - "Basketball"
    - "gym"
    - "coding":
        language:
            - Python
            - Java
        editor:
            - vscode
            - JupyterLab
university: "KIT"
student: True
```

We use `yaml.safe_load()` to read a YAML file:

```python
with open("test.yaml", "r") as f:
    person_info = yaml.safe_load(f)

person_info
```

```
{
    'age': 20,
    'hobby': [
    	'Basketball',
    	'gym',
    	{'coding': 
    		{
    			'editor': ['vscode', 'JupyterLab'],
    			'language': ['Python', 'Java']
    		}
    	}
    ],
    'name': 'Ecko',
    'student': True,
    'university': 'KIT'
}
```

The YAML file will be read to a dict, and we can access keys and values. For example,

```python
print(person_info["name"])
```

```
Ecko
```

### Read multiple YAMLs

Multiple YAML documents are read with `yaml.safe_load_all()`.

Example: *persons.yaml*

```yaml
---
name: "Ecko"
age: 20
hobby:
    - "Basketball"
    - "gym"
    - "coding":
        language:
            - Python
            - Java
        editor:
            - vscode
            - JupyterLab
university: "KIT"
student: True
---
name: "Tan"
age: 21
hobby:
    - football
    - reading
university: "TUM"
student: True
```

We read this YAML file and print name and age of each person:

```python
with open("persons.yaml") as stream:
    persons = yaml.safe_load_all(stream)

    for person in persons:
        name = person["name"]
        age = person["age"]
        print(f"Name: {name}, Age: {age}")
```

```
Name: Ecko, Age: 20
Name: Tan, Age: 21
```

### Write YAML

To serialize a Python object into a YAML stream, use `yaml.safe_dump()`.

In order to format the output YAML, you can use keyword arguments of `safe_dump()`. Some useful arguments are:

- `default_flow_style `: indicates if a collection is block or flow. The possible values are `None`, `True`, `False`.
- `indent`: sets the preferred indentation
- `explicit_start`: if `True`, adds an explicit start using `---`
- `explicit_end`: if `True`, adds an explicit end using `---`

For example

```python
person_info = {
    "name": "Ecko",
    "age": 20,
    "hobby": [
        "Basketball", 
        {
            "coding": {
                "language": ["Python", "Java", "C#"],
                "start_from": 2014
            }
        }
    ]
}

with open("person.yaml", "w") as f:
    yaml.safe_dump(person_info, f, default_flow_style=False, indent=2)
```

`person.yaml` will look like following:

```yaml
age: 20
hobby:
- Basketball
- coding:
    language:
    - Python
    - Java
    - C#
    start_from: 2014
name: Ecko
```

### Write multiple YAML files

Use `yaml.safe_dump_all()`

## Reference

- [Python YAML tutorial](http://zetcode.com/python/yaml/): short and quick tutorial
- [Python的PyYAML模块详解](https://blog.csdn.net/swinfans/article/details/88770119)

- [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation): official documentation of PyYAML. However, it is out-of-date and maybe the most messy and the worst documentation I've ever read. :thumbsdown:

