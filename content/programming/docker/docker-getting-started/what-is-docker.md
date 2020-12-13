---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 11

# Basic metadata
title: "What is Docker?"
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
        weight: 1
---

## Overview

- Docker is an open platform for developing, shipping, and running applications. 
- Docker enables you to **separate your applications from your infrastructure** so you can deliver software quickly. 
- With Docker, you can manage your infrastructure in the same ways you manage your applications. By taking advantage of Docker’s methodologies for shipping, testing, and deploying code quickly, you can significantly reduce the delay between writing code and running it in production. :clap:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*NXZYK4_f0lFJ8gpgcE5tHA.png" alt="Docker for Dummies, Literally. Yes, those days are long gone when you… | by  Saad A Akash | Medium" style="zoom: 67%;" />

## Docker architecture

### Docker engine

*Docker Engine* is a client-server application with these major components:

- A server which is a type of long-running program called a **daemon** process (the `dockerd` command).
- A REST API which specifies interfaces that programs can use to talk to the daemon and instruct it what to do.
- A command line interface (CLI) client (the `docker` command).

![Docker Engine Components Flow](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/engine-components-flow.png)



### Docker images

- An *image* is a read-only template with instructions for creating a Docker container. 
- Often, an image is *based on* another image, with some additional customization. 
  - For example, you may build an image which is based on the `ubuntu` image, but installs the Apache web server and your application, as well as the configuration details needed to make your application run.
- You might create your own images or you might only use those created by others and published in a registry. 
  - To build your own image, you create a ***Dockerfile*** with a simple syntax for defining the steps needed to create the image and run it. 
    - Each instruction in a Dockerfile creates a layer in the image. 
    - When you change the Dockerfile and rebuild the image, only those layers which have changed are rebuilt.

### Docker containers

- A container is a **runnable instance of an image**. 
- You can create, start, stop, move, or delete a container using the Docker API or CLI. You can connect a container to one or more networks, attach storage to it, or even create a new image based on its current state.
- Containerization is increasingly popular because containers are:
  - **Flexible**: Even the most complex applications can be containerized.
  - **Lightweight**: Containers leverage and share the host kernel, making them much more efficient in terms of system resources than virtual machines.
  - **Portable**: You can build locally, deploy to the cloud, and run anywhere.
  - **Loosely coupled**: Containers are highly self sufficient and encapsulated, allowing you to replace or upgrade one without disrupting others.
  - **Scalable**: You can increase and automatically distribute container replicas across a datacenter.
  - **Secure**: Containers apply aggressive constraints and isolations to processes without any configuration required on the part of the user.

### Registries

- A Docker *registry* stores Docker images. Docker Hub is a public registry that anyone can use, and Docker is configured to look for images on Docker Hub by default. You can even run your own private registry.
- When you use the `docker pull` or `docker run` commands, the required images are pulled from your configured registry. When you use the `docker push` command, your image is pushed to your configured registry.

## Docker Vs. Virtual Machines (VMs)

A container runs *natively* on Linux and shares the kernel of the host machine with other containers. It runs a discrete process, taking no more memory than any other executable, making it lightweight.

By contrast, a **virtual machine** (VM) runs a full-blown “guest” operating system with *virtual* access to host resources through a hypervisor. In general, VMs incur a lot of overhead beyond what is being consumed by your application logic.

![截屏2020-12-12 22.34.09](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-12-12%2022.34.09.png)

## Get Docker

See: [Get Docker](https://docs.docker.com/get-docker/)



## Reference

- [Docker Tutorial for Beginners](https://www.guru99.com/docker-tutorial.html)

- [Docker overview](https://docs.docker.com/get-started/overview/)