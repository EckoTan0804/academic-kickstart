---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 12

# Basic metadata
title: "Docker Container"
date: 2020-12-12
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
        weight: 2
---

## TL;DR

**Lifecycle**

- [`docker create`](https://docs.docker.com/engine/reference/commandline/create) creates a container but does not start it.
- [`docker rename`](https://docs.docker.com/engine/reference/commandline/rename/) allows the container to be renamed.
- [`docker run`](https://docs.docker.com/engine/reference/commandline/run) creates and starts a container in one operation.
- [`docker rm`](https://docs.docker.com/engine/reference/commandline/rm) deletes a container.
- [`docker update`](https://docs.docker.com/engine/reference/commandline/update/) updates a container's resource limits.

**Starting and stpping**

- [`docker start`](https://docs.docker.com/engine/reference/commandline/start) starts a container so it is running.
- [`docker stop`](https://docs.docker.com/engine/reference/commandline/stop) stops a running container.
- [`docker restart`](https://docs.docker.com/engine/reference/commandline/restart) stops and starts a container.
- [`docker pause`](https://docs.docker.com/engine/reference/commandline/pause/) pauses a running container, "freezing" it in place.
- [`docker unpause`](https://docs.docker.com/engine/reference/commandline/unpause/) will unpause a running container.
- [`docker wait`](https://docs.docker.com/engine/reference/commandline/wait) blocks until running container stops.
- [`docker kill`](https://docs.docker.com/engine/reference/commandline/kill) sends a SIGKILL to a running container.
- [`docker attach`](https://docs.docker.com/engine/reference/commandline/attach) will connect to a running container.

## What is Docker container?

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/docker-containerized-appliction-blue-border_2.png" alt="img" style="zoom: 40%;" />

- Containers are an abstraction at the app layer that packages code and dependencies together. 
- Multiple containers can run on the same machine and share the OS kernel with other containers, each running as isolated processes in user space. 
- Containers take up less space than VMs (container images are typically tens of MBs in size), can handle more applications and require fewer VMs and Operating systems.

## Basic Usages

### Pulling image

We can use `docker pull` to pull images from registry

E.g.: Pull `ubuntu` image

```bash
docker pull ubuntu
```

### Running a ontainer

Running of containers is managed with the Docker `run` command. 

For example, we creat a container using an `ubuntu:15.10` image and start it:

```bash
docker run -it ubuntu:15.10 /bin/bash
```

- `-i` : interactive
- `-t ` : terminal
- `ubuntu:15.10`: ubuntu image (version tag `15.10`)
- `/bin/bash` : command behind image

If we want our docker service to run in background, we can use argument `-d`. For example:

```bash
docker run -itd --name ubuntu-test-1 ubuntu /bin/bash
```

- `--name`: assign a name `ubuntu-test-1` to this container
- `-d`: If we add this argument, it won't enter the container by default

### Listing containers

Listing running containers:

```bash
docker ps
```

Listing all containers

```bash
docker ps -a
```

### Starting a container

```bash
docker start <stopped-container-ID>=	
```

### Ruuning a command in a running container

```bash
docker exec <container-name> <command>
```

For example, we run a container firstly in background 

```bash
docker run -itd --name ubuntu-test-1 ubuntu /bin/bash
```

Check the running container:

```bash
docker ps
```

```
CONTAINER ID   IMAGE     COMMAND       CREATED          STATUS          PORTS     NAMES
67cbb5de95c2   ubuntu    "/bin/bash"   10 minutes ago   Up 10 minutes             ubuntu-test-1
```

Now we want to print `Hello world` in the terminal of container `ubuntu-test-1` :

```bash
docker exec -it ubuntu-test-1 echo "Hello World"
```

```bash
Hello World
```

### Stopping a container

```bash
docker stop <container-ID>
```

### Removing a container

```bash
docker rm -f <container-ID>
```



## Reference

- [wsargent](https://github.com/wsargent)/**[docker-cheat-sheet](https://github.com/wsargent/docker-cheat-sheet)**
- [Docker容器使用](https://www.runoob.com/docker/docker-container-usage.html)