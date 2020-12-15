---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 15

# Basic metadata
title: "Dockerfile Best Practice"
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
        weight: 5
---

## General Guidelines

### Write `.dockerignore`

- When building an image, Docker has to prepare `context` first - gather all files that may be used in a process.

- Default context contains all files in a Dockerfile directory. Usually **we don't want to include there `.git` directory,  downloaded libraries,  and compiled files**.

- Similar to `.gitignore`, `.dockerignore` can look like followings:

  *.dockerignore*

  ```
  .git/
  node_modules/
  dist/
  ```

### Container should do one thing (Decouple applications)

Technically, you can start multiple processes (such as database, frontend, backend applications) inside Docker container. However, such a big container will bite you

- long build times (change in e.g. frontend will force the whole backend to rebuild)
- very large images
- hard logging from many applications (no more simple stdout)
- wasteful horizontal scaling
- problems with zombie processes - you have to remember about proper init process

The proper way is

- prepare separate Docker image for each component
- use [Docker Compose](https://docs.docker.com/compose/) to easily start multiple containers at the same time.

### Minimize the number of layers

Docker is all about **layers**. 

- Each command in Dockerfile creates so-called *layer*
- Layers are cached and reused
- Invalidating cache of a single layer invalidates all subsequent layers
- Invalidation occurs after command change, if copied files are different, or build variable is other than previously
- Layers are immutable, so if we add a file in one layer, and remove it in the next one, image STILL contains that file (it's just not available in the container)!

Minimizing the number of steps in your image may improve build and pull performance. Therefore it’s a cool best practice to combine several steps into one line, so that they’ll create only one intermediary image.

{{% alert note %}} 

Only `RUN`, `COPY` and `ADD` instructions create layers to improve build performance. Other instructions create temporary intermediate images, and do not increase the size of the build.

{{% /alert %}} 

### Example

❌

```dockerfile
FROM alpine:3.4

RUN apk update
RUN apk add curl
RUN apk add vim
RUN apk add git
```

✅

```dockerfile
FROM alpine:3.4

RUN apk update && \
    apk add curl && \
    apk add vim && \
    apk add git
```

After building this Dockerfile the usual way you’ll find that this time it has only taken 2 steps instead of 5.

### Sort multi-line arguments

- Whenever possible, ease later changes by **sorting multi-line arguments alphanumerically**.

- This helps to avoid duplication of packages and make the list much easier to update. This also makes PRs a lot easier to read and review. Adding a space before a backslash (`\`) helps as well.

Example

```dockerfile
RUN apt-get update && apt-get install -y \
  bzr \
  cvs \
  git \
  mercurial \
  subversion \
  && rm -rf /var/lib/apt/lists/*
```



### Do not use 'latest' base image tag

- `latest` tag is a default one, used when no other tag is specified. E.g. our instruction `FROM ubuntu` in reality does exactly the same as `FROM ubuntu:latest`

- But 'latest' tag will point to a different image when a new version will be released, and your build may break. :cry:
- So, unless you are creating a generic Dockerfile that must stay up-to-date with the base image, **provide specific tag**!!!

### Remove unneeded files after each RUN step

Let's assume we updated apt-get sources, installed few packages required for compiling others, downloaded and extracted archives. We obviously don't need them in our final images, so better let's make a cleanup.

E.g. we can remove apt-get lists (created by `apt-get update`):

```dockerfile
FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y nodejs \
    # added lines
    && rm -rf /var/lib/apt/lists/*

ADD . /app
RUN cd /app && npm install

CMD npm start
```

### Use proper base image

Use specilaized image instead of general-purpose base image. For example, if we just want to run node application, instead of using `ubuntu` as our base image, we should use `node` (or even alpine version).

### Set `WORKDIR` and `CMD`

- `WORKDIR` command changes default directory, where we run our `RUN` / `CMD` / `ENTRYPOINT` commands.
- `CMD` is a default command run after creating container without other command specified. It's usually the most frequently performed action. 

Example

```dockerfile
FROM node:7-alpine

WORKDIR /app

ADD . /app

RUN npm install

CMD ["npm", "start"]
```

### Use `ENTRYPOINT` (optional)

### Use "exec" inside entrypoint script

### Prefer `COPY` over `ADD`

- `COPY` is simpler.
- `ADD` has some logic for downloading remote files and extracting archives (more see [official documentation](https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/#add-or-copy))
- **Just stick with `COPY`** !:muscle:

### Use multi-stage builds

- [Multi-stage builds](https://docs.docker.com/develop/develop-images/multistage-build/) allow you to drastically reduce the size of your final image, without struggling to reduce the number of intermediate layers and files.
- Because an image is built during the final stage of the build process, you can minimize image layers by [leveraging build cache](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cache).
- If your build contains several layers, you can order them **from the less frequently changed** (to ensure the build cache is reusable) **to the more frequently changed**:
  - Install tools you need to build your application
  - Install or update library dependencies
  - Generate your application

### Leverage build cache

- When building an image, Docker steps through the instructions in your `Dockerfile`, executing each in the order specified. As each instruction is examined, Docker looks for an existing image in its cache that it can reuse, rather than creating a new (duplicate) image.
- The basic rules that Docker follows are outlined below:
  - Starting with a parent image that is already in the cache, the next instruction is compared against all child images derived from that base image to see if one of them was built using the exact same instruction. If not, the cache is invalidated.
  - In most cases, simply comparing the instruction in the `Dockerfile` with one of the child images is sufficient. 
  - For the `ADD` and `COPY` instructions, the contents of the file(s) in the image are examined and a checksum is calculated for each file. The last-modified and last-accessed times of the file(s) are not considered in these checksums. During the cache lookup, the checksum is compared against the checksum in the existing images. If anything has changed in the file(s), such as the contents and metadata, then the cache is invalidated.
  - Aside from the `ADD` and `COPY` commands, cache checking does not look at the files in the container to determine a cache match. 
  - Once the cache is invalidated, all subsequent `Dockerfile` commands generate new images and the cache is not used.

### Specify default environment variables, ports and volumes

It's a good practice to set default values in Dockerfile, which can make our dockerfile more consistent and flexible.

Example

```dockerfile
FROM node:7-alpine

# ENV variables required during build
ENV PROJECT_DIR=/app

WORKDIR $PROJECT_DIR

COPY package.json $PROJECT_DIR
RUN npm install
COPY . $PROJECT_DIR

ENV MEDIA_DIR=/media \
    NODE_ENV=production \
    APP_PORT=3000
    
VOLUME $MEDIA_DIR
EXPOSE $APP_PORT
```

### Add metadata to image using LABEL

### Create ephemeral containers

The image defined by your `Dockerfile` should generate containers that are as ephemeral as possible. By “ephemeral”, we mean that the container can be stopped and destroyed, then rebuilt and replaced with an absolute minimum set up and configuration.

### Don’t install unnecessary packages

To reduce complexity, dependencies, file sizes, and build times, avoid installing extra or unnecessary packages just because they might be “nice to have.” 



## Dockerfile instructions

### FROM

- Whenever possible, use current official images as the basis for your images.
- We recommend the [Alpine image](https://hub.docker.com/_/alpine/) as it is tightly controlled and small in size (currently under 5 MB), while still being a full Linux distribution.

### LABEL

- You can add labels to your image to help organize images by project, record licensing information, to aid in automation, or for other reasons.  (More see: [Understanding object labels](https://docs.docker.com/config/labels-custom-metadata/))

- Acceptable formats

  - One label per line

    ```dockerfile
    LABEL com.example.version="0.0.1-beta"
    LABEL com.example.release-date="2015-02-12"
    ```

  - Multiple labels on one line

    ```dockerfile
    LABEL com.example.version="0.0.1-beta" com.example.release-date="2015-02-12"
    ```

  - Set multiple labels at once, using line-continuation characters to break long lines

    ```dockerfile
    LABEL com.example.version="0.0.1-beta" \
    	  com.example.release-date="2015-02-12"
    ```

### RUN

Split long or complex `RUN` statements on multiple lines separated with backslashes to make your `Dockerfile` more readable, understandable, and maintainable.

#### `apt-get`

- Avoid `RUN apt-get upgrade` and `dist-upgrade`

- Always combine `RUN apt-get update` with `apt-get install` in the same `RUN` statement. This ensures your Dockerfile installs the latest package versions with no further coding or manual intervention. (“cache busting”)

  E.g.

  ```dockerfile
  RUN apt-get update && apt-get install -y \
      package-bar \
      package-baz \
      package-foo  \
      && rm -rf /var/lib/apt/lists/*
  ```

- You can also achieve cache-busting by specifying a package version. ("cache busting")

  E.g.

  ```dockerfile
  RUN apt-get update && apt-get install -y \
      package-bar \
      package-baz \
      package-foo=1.3.*
  ```

Below is a well-formed `RUN` instruction that demonstrates all the `apt-get` recommendations.

```dockerfile
RUN apt-get update && apt-get install -y \
    aufs-tools \
    automake \
    build-essential \
    curl \
    dpkg-sig \
    libcap-dev \
    libsqlite3-dev \
    mercurial \
    reprepro \
    ruby1.9.1 \
    ruby1.9.1-dev \
    s3cmd=1.1.* \
 && rm -rf /var/lib/apt/lists/*
```

### CMD

- The `CMD` instruction should be used to run the software contained in your image, along with any arguments.
- `CMD` should almost always be used in the form of `CMD ["executable", "param1", "param2"…]`.
- `CMD` should rarely be used in the manner of `CMD ["param", "param"]` in conjunction with [`ENTRYPOINT`](https://docs.docker.com/engine/reference/builder/#entrypoint), unless you and your expected users are already quite familiar with how `ENTRYPOINT` works.

### EXPOSE

- The `EXPOSE` instruction indicates the ports on which a container listens for connections.
- You should use the common, traditional port for your application. E.g.
  -  an image containing the Apache web server would use `EXPOSE 80`
  - an image containing MongoDB would use `EXPOSE 27017`

### ENV

- To make new software easier to run, you can use `ENV` to update the `PATH` environment variable for the software your container installs.

- `ENV` instruction is also useful for providing required environment variables specific to services you wish to containerize

- `ENV` can also be used to set commonly used version numbers so that version bumps are easier to maintain

  E.g.

  ```dockerfile
  ENV PG_MAJOR=9.3
  ENV PG_VERSION=9.3.4
  RUN curl -SL https://example.com/postgres-$PG_VERSION.tar.xz | tar -xJC /usr/src/postgress && …
  ENV PATH=/usr/local/postgres-$PG_MAJOR/bin:$PATH
  ```

- Each `ENV` line creates a new intermediate layer, just like `RUN` commands

  - This means that even if you unset the environment variable in a future layer, it still persists in this layer and its value can’t be dumped. 
  - You can separate your commands with `;` or `&&`. Using `\` as a line continuation character for Linux Dockerfiles improves readability.
  - Or you could also put all of the commands into a shell script and have the `RUN` command just run that shell script.

### ADD or COPY

- **`COPY` is preferred.** 

- If you have multiple `Dockerfile` steps that use different files from your context, **`COPY` them individually, rather than all at once.** This ensures that each step’s build cache is only invalidated (forcing the step to be re-run) if the specifically required files change.

  E.g.

  ```dockerfile
  COPY requirements.txt /tmp/
  RUN pip install --requirement /tmp/requirements.txt
  COPY . /tmp/
  ```

  This results in fewer cache invalidations for the `RUN` step, than if you put the `COPY . /tmp/` before it.

- Using `ADD` to fetch packages from remote URLs is **strongly discouraged** 🙅‍♂️; you should use `curl` or `wget` instead.

  - That way you can delete the files you no longer need after they’ve been extracted and you don’t have to add another layer in your image.

  - E.g.

    ❌

    ```dockerfile
    ADD https://example.com/big.tar.xz /usr/src/things/
    RUN tar -xJf /usr/src/things/big.tar.xz -C /usr/src/things
    RUN make -C /usr/src/things all
    ```

    ✅

    ```dockerfile
    RUN mkdir -p /usr/src/things \
        && curl -SL https://example.com/big.tar.xz \
        | tar -xJC /usr/src/things \
        && make -C /usr/src/things all
    ```

- For other items (files, directories) that do not require `ADD`’s tar auto-extraction capability, you should always use `COPY`.

### ENTRYPOINT

- The best use for `ENTRYPOINT` is to set the image’s main command, allowing that image to be run as though it was that command (and then use `CMD` as the default flags).

  - Example: image for the command line tool `s3cmd`

    ```dockerfile
    ENTRYPOINT ["s3cmd"]
    CMD ["--help"]
    ```

    Now the image can be run like this to show the command’s help:

    ```bash
    docker run s3cmd
    ```

    or use the right parameters to execute a command:

    ```bash
    docker run s3cmd ls s3://mybucket
    ```

- The `ENTRYPOINT` instruction can also be used in combination with a helper script, allowing it to function in a similar way to the command above, even when starting the tool may require more than one step.

### WORKDIR

- For clarity and reliability, you should always use absolute paths for your `WORKDIR`.
- You should use `WORKDIR` instead of proliferating instructions like `RUN cd … && do-something`, which are hard to read, troubleshoot, and maintain.

## Reference

- {{< icon name="docker" pack="fab" >}} Official guide: [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [How to write excellent Dockerfiles](https://rock-it.pl/how-to-write-excellent-dockerfiles/)