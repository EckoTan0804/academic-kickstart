---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 203

# Basic metadata
title: "pre-commit"
date: 2021-11-01
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Best practice"]
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
        parent: py-best-practice
        weight: 3
---

## TL;DR

Before committing any staged Python files, `pre-commit` automatically formats the code, validates compliance to PEP8, and performs different types of checking to keep the code and the project clean. This automatical process can greatly save our time on code formatting so that we can concentrate on code logic.

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/precommit_pipeline-20211101151024237-20211101153634949.png" caption="Basic `pre-commit` workflow (source: [Automate Python workflow using pre-commits: black and flake8](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/))" numbered="true" >}}

## What is `pre-commit`?

`pre-commit` is a multi-language package manager for pre-commit hooks. You specify a list of hooks you want and pre-commit manages the installation and execution of any hook written in any language before every commit.

## Quick start

1. Install pre-commit package manager

   - pip

     ```bash
     pip install pre-commit
     ```

   - conda

     ```bash
     conda install -c conda-forge pre-commit
     ```

   Check if the installation is successful:

   ```bash
   pre-commit --version
   ```

   If successful, you can see the version information 

2. (Optional) Add `pre-commit` to `requirements.txt`

3. Add a pre-commit configuration

   1. Create a file named `.pre-commit-config.yaml` in the root of the project

   2. Define hooks/plugins in `.pre-commit-config.yaml` (More see: [Adding pre-commit plugins](#adding-pre-commit-plugins))

      > You can generate a very basic configuration using [`pre-commit sample-config`](https://pre-commit.com/#pre-commit-sample-config)

4. Install the git hook scripts (prerequisite: git is already initialized)

   ```bash
   pre-commit install
   ```

   Now `pre-commit` will run automatically on every `git commit` (usually `pre-commit` will only run on the changed files during git hooks).

5. (Optional) Run `pre-commit` against all the files

   ```bash
   pre-commit run --all-files
   ```

   

## Adding pre-commit plugins

Once you have `pre-commit` installed, adding pre-commit hooks/plugins to your project is done with the `.pre-commit-config.yaml` configuration file, which describes what repositories and hooks are installed.

The top-level of `.pre-commit-config.yaml` is a map. Among them, the most important key is `repos`, which is a list of repository mappings. For other keys see: [.pre-commit-config.yaml - top level](https://pre-commit.com/#pre-commit-configyaml---top-level).

### `repos`

The repository mapping tells pre-commit where to get the code for the hook from.

| [`repo`](https://pre-commit.com/#repos-repo)   | the repository url to `git clone` from                       |
| ---------------------------------------------- | ------------------------------------------------------------ |
| [`rev`](https://pre-commit.com/#repos-rev)     | the revision or tag to clone at. (*new in 1.7.0*: previously `sha`) |
| [`hooks`](https://pre-commit.com/#repos-hooks) | A list of [hook mappings](https://pre-commit.com/#pre-commit-configyaml---hooks). |

Example

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v1.2.3
    hooks:
    -   ...
```

#### `hooks`

The [hook mapping](https://pre-commit.com/#pre-commit-configyaml---hooks) configures which hook from the repository is used and allows for customization. 

- The necessary key is `id`, telling which hook from the repository to use. Other keys are optional.

- All optional keys will receive their default from the repository's configuration.

Example

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v1.2.3
    hooks:
    -   id: trailing-whitespace
```



## Useful repos and hooks

We briefly introduce some important and useful hooks. For supported hooks, check the [website](https://pre-commit.com/hooks.html) of `pre-commit`.

### `pre-commit-hooks`

Some out-of-the-box hooks for pre-commit.

Example

```yaml
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.0.1
    hooks:
      - id: check-added-large-files # Prevent giant files from being committed.
      - id: check-ast # Simply check whether files parse as valid python.
      - id: check-byte-order-marker
      - id: check-case-conflict # Check for files with names that would conflict on a case-insensitive filesystem like MacOS HFS+ or Windows FAT
      - id: check-docstring-first # Checks for a common error of placing code before the docstring.
      - id: check-json # Attempts to load all json files to verify syntax.
      - id: check-yaml # Attempts to load all yaml files to verify syntax.
      - id: end-of-file-fixer # Makes sure files end in a newline and only a newline.
      - id: trailing-whitespace # Trims trailing whitespace.
      - id: mixed-line-ending # Replaces or checks mixed line ending.
```

### black

The [black](https://github.com/psf/black) code formatter in Python is an "uncompromising" tool that formats your code in the best way possible. 

Example

```yaml
  - repo: https://github.com/psf/black
    rev: 20.8b1
    hooks:
      - id: black
        args:
          - --line-length=119
```

### isort

[isort](https://pycqa.github.io/isort/index.html) is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type.

To allow customizations to be integrated into any project quickly, isort supports various standard config formats. Check the [documentation](https://pycqa.github.io/isort/docs/configuration/config_files.html) for all supported formats. I personally prefer `setup.cfg`. 

> When applying configurations, isort looks for the closest supported config file, in the order files are listed below. You can manually specify the settings file or path by setting `--settings-path` from the command-line. Otherwise, isort will traverse up to 25 parent directories until it finds a suitable config file.

For projects that officially use both isort and [black](https://github.com/psf/black), it is recommended to set the black profile in a config file.

Example: `setup.cfg`

```toml
[isort]
multi_line_output = 3
include_trailing_comma = True
force_grid_wrap = 0
use_parentheses = True
line_length = 88
profile = black
```

{{% alert note%}} 

Sometimes the skip options do not work well, to skip some file for some reasons, check: [Action Comments](https://pycqa.github.io/isort/docs/configuration/action_comments.html).

{{% /alert%}}

### flake8

[flake8](https://github.com/PyCQA/flake8) is a command-line utility for enforcing style consistency across Python projects. It is a wrapper around these tools:

- PyFlakes
- pycodestyle
- Ned Batchelder's McCabe script.

A good pratice to customized flake8 checking is to define custom configurations in `setup.cfg` and specify its path with `--args` in `.pre-commit-config.yaml`:

`setup.cfg`

```toml
[flake8]
max-line-length = 119
max-complexity = 11
ignore = C901, W503, W504, E203, F401

[isort]
multi_line_output = 3
include_trailing_comma = True
force_grid_wrap = 0
use_parentheses = True
line_length = 88
profile = black
```

and in `.pre-commit-config.yaml`

```yaml
  - repo: https://github.com/PyCQA/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args:
          - --config=setup.cfg
```



## GitHub Gist

My [personal pre-commit config](https://gist.github.com/EckoTan0804/8d08e4cea323644dd82c43190a55eaa7) for python projects.







## Reference

- [Documentation of pre-commit](https://pre-commit.com/#usage)

- [Automate Python workflow using pre-commits: black and flake8](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/): a simple tutorial on pre-commit
- [Running Python Linters with Pre-commit Hooks](https://rednafi.github.io/digressions/python/2020/04/06/python-precommit.html): Another tutorial on pre-commit
- [Python教程:如何建立完美自动化的Python-starter项目](https://mojotv.cn/tutorial/how-get-setup-perfect-python-projetc)
- [Python Pre-Commit Hooks Setup in a single video!](https://www.youtube.com/watch?v=Wmw-VGSjSNg&ab_channel=SoftwareEngineerHaydn): Video tutorial
- Some good examples of `.pre-commit-config.yaml`
  - https://github.com/NCAS-CMS/cfunits/blob/master/.pre-commit-config.yaml
  - https://github.com/open-mmlab/mmsegmentation/blob/master/.pre-commit-config.yaml: `.pre-commit-config.yaml` of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

