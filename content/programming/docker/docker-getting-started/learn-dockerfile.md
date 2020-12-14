---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 14

# Basic metadata
title: "Dockerfile"
date: 2020-12-13
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Docker"]
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
    docker:
        parent: docker-getting-started
        weight: 4
---

## TL;DR



## What is Dockerfile?

**A Dockerfile is a text file that defines a Docker [image]({{< relref "docker-image.md" >}}). You’ll use a Dockerfile to create your own custom Docker image, in other words to define your custom environment to be used in a Docker [container]({{< relref "docker-container.md" >}}).**

![Build a Docker Image just like how you would configure a VM | by Nilesh  Jayanandana | Platformer Cloud | Medium](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*p8k1b2DZTQEW_yf0hYniXw.png)

A Dockerfile is a step by step definition of building up a Docker image. The Dockerfile contains a list of instructions that Docker will execute when you issue the `docker build` command. Your workflow is like this:

1. Create the **Dockerfile** and define the steps that build up your images
2. Issue the `docker build` command which will build a Docker image from your Dockerfile
3. Use this image to start containers with the `docker run` command

## Dockerfile commands

### FROM

- The `FROM` instruction initializes a new build stage and sets the [*Base Image*](https://docs.docker.com/glossary/#base_image) for subsequent instructions. 

- As such, a valid `Dockerfile` must start with a `FROM` instruction. 

  - Note: `ARG` is the only instruction that may precede `FROM` in the `Dockerfile`.

- The image can be any valid image – it is especially easy to start by **pulling an image** from the [*Public Repositories*](https://docs.docker.com/docker-hub/repos/).

- Syntax:

  ```dockerfile
  FROM [--platform=<platform>] <image>[:<tag>] [AS <name>]
  ```

- Example:

  ```dockerfile
  FROM ubuntu:18.04
  ```

### RUN

- Execute a command in a shell or exec form. 

- The RUN instruction **adds a new layer** on top of the newly created image. 

- The committed results are then used for the next instruction in the DockerFile.

- Syntax

  - Shell form (the command is run in a shell, which by default is `/bin/sh -c` on Linux)

    ```dockerfile
    RUN <command>
    ```

    In the *shell* form you can use a `\` (backslash) to continue a single RUN instruction onto the next line.

    E.g.

    ```dockerfile
    RUN /bin/bash -c 'source $HOME/.bashrc; \
    echo $HOME'
    ```

    Together they are equivalent to this single line:

    ```dockerfile
    RUN /bin/bash -c 'source $HOME/.bashrc; echo $HOME'
    ```

  - Exec form

    ```dockerfile
    RUN ["executable", "param1", "param2"]
    ```

    E.g.

    ```dockerfile
    RUN ["/bin/bash", "-c", "echo hello"]
    ```

### CMD

- Give the default commands **when the image is instantiated**, it doesn’t execute while build stage.

- The `CMD` instruction has three forms:
  - `CMD ["executable","param1","param2"]` (***exec*** form, this is the **preferred** form)

    - It will NOT involke a command shell.  This means that normal shell processing does NOT happen.

    E.g 

    ```dockerfile
    CMD ["echo","hello", "world"]
    ```

  - `CMD ["param1","param2"]` (as *default parameters to ENTRYPOINT*)

    - If you would like your container to run the same executable every time, then you should consider using `ENTRYPOINT` in combination with `CMD`. See [*ENTRYPOINT*](https://docs.docker.com/engine/reference/builder/#entrypoint).

  - `CMD command param1 param2` (***shell*** form)

    - The `<command>` will execute in `/bin/sh -c`. 

    - E.g.

      ```dockerfile
      CMD echo $HW
      ```

      It will execute the command `/bin/sh -c echo $HW`

- There can only be **ONE** `CMD` instruction in a `Dockerfile`. If you list more than one `CMD` then only the last `CMD` will take effect.

- **The main purpose of a `CMD` is to provide defaults for an executing container.**  These defaults can include 

  - an executable, or
  - they can omit the executable, in which case you must specify an `ENTRYPOINT` instruction as well.

{{% alert warning %}} 

If the user specifies arguments to `docker run` then they will override the default specified in `CMD`.

{{% /alert %}}

### ENTRYPOINT

- Configures a container that will run as an executable. I.e., a specific application can be set as default and run every time a container is created using the image.

- `ENTRYPOINT` has two forms:

  - *exec* form (the preferred form)

    ```dockerfile
    ENTRYPOINT ["executable", "param1", "param2"]
    ```

  - *shell* form:

    ```dockerfile
    ENTRYPOINT command param1 param2
    ```

- Unlike `CMD`, command line arguments to `docker run <image>` will NOT override the default specified in `CMD`. Instead, they will be **appended** after all elements in an *exec* form `ENTRYPOINT`, and will override all elements specified using `CMD`. 
- Only the last `ENTRYPOINT` instruction in the `Dockerfile` will have an effect.
- You can override the `ENTRYPOINT` instruction using the `docker run --entrypoint` flag.

#### Understand how CMD and ENTRYPOINT interact

Both `CMD` and `ENTRYPOINT` instructions define what command gets executed when running a container. There are few rules that describe their co-operation.

1. Dockerfile should specify at least one of `CMD` or `ENTRYPOINT` commands.
2. `ENTRYPOINT` should be defined when using the container as an executable.
3. `CMD` should be used as a way of defining default arguments for an `ENTRYPOINT` command or for executing an ad-hoc command in a container.
4. `CMD` will be overridden when running the container with alternative arguments.

The table below shows what command is executed for different `ENTRYPOINT` / `CMD` combinations:

|                                | No ENTRYPOINT                | ENTRYPOINT exec_entry p1_entry   | ENTRYPOINT [“exec_entry”, “p1_entry”]            |
| :----------------------------- | :--------------------------- | :------------------------------- | :----------------------------------------------- |
| **No CMD**                     | *error, not allowed*         | `/bin/sh -c exec_entry p1_entry` | `exec_entry p1_entry`                            |
| **CMD [“exec_cmd”, “p1_cmd”]** | `exec_cmd p1_cmd`            | `/bin/sh -c exec_entry p1_entry` | `exec_entry p1_entry exec_cmd p1_cmd`            |
| **CMD [“p1_cmd”, “p2_cmd”]**   | `p1_cmd p2_cmd`              | `/bin/sh -c exec_entry p1_entry` | `exec_entry p1_entry p1_cmd p2_cmd`              |
| **CMD exec_cmd p1_cmd**        | `/bin/sh -c exec_cmd p1_cmd` | `/bin/sh -c exec_entry p1_entry` | `exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd` |

A common use case is: the **executable is defined with ENTRYPOINT**, while **CMD specifies the default parameter**.

Example:

*Dockerfile*

```dockerfile
FROM node:8.11-slim

CMD ["world"]

ENTRYPOINT ["echo", "Hello"]
```

Build image:

```bash
docker built -t my-docker-image .
```

Run container:

- without argument

  ```bash
  docker run -it my-docker-image
  ```

  output:

  ```
  Hello world
  ```

  > If we run `docker run` without argument, the default argument `world` defined in `CMD` will be used. I.e, the command `echo Hello World` will be executed.

- with argument

  ```
  docker run -it my-docker-image James Bond
  ```

  output:

  ```
  Hello James Bond
  ```

  > If we specify arguments when running `docker run`, `CMD` will be overriden by our given arguments. In our case, `CMD ["World"]` is overriden to `CMD ["James", "Bond"]` and then applied to `echo`. Thus, the command `echo Hello James Bond` will be executed.

> Summary of `CMD` and `ENTRYPOINT` see: [Docker CMD Vs Entrypoint Commands: What's The Difference?](https://phoenixnap.com/kb/docker-cmd-vs-entrypoint)

### EXPOSE

- Specify the port on which the container will be listening at runtime by running the EXPOSE instruction.

- Syntax

  ```dockerfile
  EXPOSE <port>
  ```

### ENV

- Set environment variables using the ENV instruction. 
- This value will be in the environment for all subsequent instructions in the build stage and can be [replaced inline](https://docs.docker.com/engine/reference/builder/#environment-replacement) in many as well.
- They come as key value pairs and increases the flexibility of running programs.

- Syntax

  ```dockerfile
  ENV <key>=<value> ...
  ```

  e.g.

  ```dockerfile
  ENV PATH=usr/node
  ```

  or 

  ```dockerfile
  ENV <key> <value>
  ```

  e.g.

  ```dockerfile
  ENV PATH usr/node
  ```

### ADD

- `ADD` has two forms:

  ```dockerfile
  ADD [--chown=<user>:<group>] <src>... <dest>
  ```

  or

  ```dockerfile
  ADD [--chown=<user>:<group>] ["<src>",... "<dest>"]
  ```

- Copies new files, directories or remote file URLs from `<src>` and adds them to the filesystem of the image at the path `<dest>`.
- Invalidates caches. **Avoid `ADD` and use `COPY` instead.**

### COPY

- Syntax

  ```dockerfile
  COPY [--chown=<user>:<group>] <src>... <dest>
  ```

  or 

  ```dockerfile
  COPY [--chown=<user>:<group>] ["<src>",... "<dest>"]
  ```

- Copy new files or directories from `<src>`  and adds them to the filesystem of the container at the path `<dest>`.

  - Each `<src>` may contain wildcards and matching will be done using Go’s [filepath.Match](http://golang.org/pkg/path/filepath#Match) rules.`COPY` obeys the following rules:

- Example: copy `test.txt` to `<WORKDIR>/relativeDir/`:

  ```dockerfile
  COPY test.txt relativeDir/
  ```

- `COPY` obeys the following rules:

  - The `<src>` path must be inside the *context* of the build

  - If `<src>` is a directory, the entire contents of the directory are copied, including filesystem metadata.

    (The directory itself is not copied, just its contents.)

  - If `<src>` is any other kind of file, it is copied individually along with its metadata. In this case, if `<dest>` ends with a trailing slash `/`, it will be considered a directory and the contents of `<src>` will be written at `<dest>/base(<src>)`.

  - If multiple `<src>` resources are specified, either directly or due to the use of a wildcard, then `<dest>` must be a directory, and it must end with a slash `/`.

  - If `<dest>` does not end with a trailing slash, it will be considered a regular file and the contents of `<src>` will be written at `<dest>`.

  - If `<dest>` doesn’t exist, it is created along with all missing directories in its path.

### VOLUMN

- Creates a mount point for externally mounted volumes or other containers

### WORKDIR

- Sets the working directory for any `RUN`, `CMD`, `ENTRYPOINT`, `COPY` and `ADD` instructions that follow it in the `Dockerfile`.

- The `WORKDIR` instruction can be used multiple times in a `Dockerfile`. If a relative path is provided, it will be relative to the path of the previous `WORKDIR` instruction. E.g.

  ```dockerfile
  WORKDIR /usr/node
  WORKDIR app
  ```

  The resulted work directory will be `/usr/mode/app`

- The `WORKDIR` instruction can resolve environment variables previously set using `ENV`.  E.g.

  ```dockerfile
  ENV WORK_DIR=/usr/node/app
  
  WORKDIR ${WORK_DIR}
  ```

### ARG

- Defines a build-time variable

- Syntax

  ```dockerfile
  ARG <name>[=<default value>]
  ```

- `ARG` is the only instruction that can occur before the `FROM`.

  E.g. we can use `ARG` to specify the version of base image

  ```dockerfile
  ARG NODE_VERSION=8.11-slim
  
  FROM node:${NODE_VERSION}
  ```

### ONBUILD

- Adds a trigger instruction when the image is used as the base for another build

### STOPSIGNAL

- Sets the system call signal that will be sent to the container to exit.

### LABEL

- Add key/value metadata to your images, containers, or daemons.

- Syntax

  ```dockerfile
  LABEL <key>=<value> <key>=<value> <key>=<value> ...
  ```

- Example

  ```dockerfile
  LABEL version="1.0"
  LABEL description="Just a demo"
  ```

- If a label already exists but with a different value, the most-recently-applied value overrides any previously-set value.

- To view an image’s labels, use the `docker image inspect` command. You can use the `--format` option to show just the labels

  ```bash
  docker image inspect --format='' <image>
  ```

### SHELL

- Override default shell is used by docker to run commands.

### HEALTHCHECK

- Tells docker how to test a container to check that it is still working.

## Example with common commands

```dockerfile
# ARG is used to pass some arguments to consecutive instructions
# this is only command other than a comment can be used before FROM.
ARG NODE_VERSION=8.11-slim

# from base image node
FROM node:${NODE_VERSION}

# LABEL add metadata to your images, containers, or daemons
LABEL "about"="This file is just an example to demostrate the basic usage of dockerfile commands"

# ENV sets the environment variables for the subsequent instructions in the build stage
ENV WORK_DIR /usr/node

# WORKDIR sets the working directory for all the consecutive commands.
# We can have multiple WORKDIR commands and will be appended with a relative path.
# E.g. we have two WORKDIR commands leads to /usr/node/app
WORKDIR ${WORK_DIR}
WORKDIR app

# VOLUME is used to create a mount point with the specified name
RUN mkdir /dockerexample
VOLUME /dockerexample

# COPY is used to copy files or directories 
# from source host filesystem to a destination in the container file system.
# Here we're gonna copy package.json from our system to container file system
COPY package.json .

# RUN executes the instructions in a new layer on top of the existing image and commit those layers
# The resulted layer will be used for the next instructions in the Dockerfile
RUN ls -ll && \
    npm install

# USER instruction sets the user name and optionally the user group to use
# when running the image and for any instructions that follow it in the Dockerfile
RUN useradd ecko
USER ecko

# ADD is used to add files or directories and remote files 
# from URL from source host filesystem to a destination in the container file system.
# Avoid ADD and use COPY instead!
ADD index.js .

# CMD command is used to give the default commands when the image is instantiated, 
# it doesn’t execute while build stage. 
# There should be only ONE CMD per Dockerfile, 
# you can list multiple but the last one will be executed.
CMD ["echo", "Hello World"]

# EXPOSE informs Docker that 
# the container listens on the specified network ports at runtime. 
EXPOSE 3070

# ENTRYPOINT is used as an executable for the container.
# We can use ENTRYPOINT for executable command 
# and use CMD command to pass some default commands to the executable.
ENTRYPOINT ["echo", "Hello"]

```

## Reference

- Tutorials
  - :fire: Step by step tutorial with example: [Docker — A Beginner’s guide to Dockerfile with a sample project](https://medium.com/bb-tutorials-and-thoughts/docker-a-beginners-guide-to-dockerfile-with-a-sample-project-6c1ac1f17490)
  - [Dockerfile tutorial by example - basics and best practices [2018]](https://takacsmark.com/dockerfile-tutorial-by-example-dockerfile-best-practices-2018/#overview)
  - [Docker Dockerfile](https://www.runoob.com/docker/docker-dockerfile.html)
  - [Docker Tutorial Series, Part 3: Automation is the Word Using DockerFile](https://www.flux7.com/tutorial/docker-tutorial-series-part-3-automation-is-the-word-using-dockerfile/)
- {{< icon name="docker" pack="fab" >}} Reference: [Dockerfile reference](https://docs.docker.com/engine/reference/builder/#entrypoint)
- [Cheatsheet](https://github.com/wsargent/docker-cheat-sheet#dockerfile)

