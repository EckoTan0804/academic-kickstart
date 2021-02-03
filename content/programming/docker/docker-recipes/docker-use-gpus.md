---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 31

# Basic metadata
title: "Use GPU within a Docker Container"
date: 2021-02-02
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
        parent: docker-recipes
        weight: 1
---

## [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)

The NVIDIA Container Toolkit allows users to build and run GPU accelerated containers. The toolkit includes a container runtime [library](https://github.com/NVIDIA/libnvidia-container) and utilities to automatically configure containers to leverage NVIDIA GPUs.

[![https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/5b208976-b632-11e5-8406-38d379ec46aa.png)](https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png)

Essentially, the NVIDIA Container Toolkit is a docker image that provides support to automatically recognize GPU drivers on your base machine and pass those same drivers to your Docker container when it runs. So if you are able to run **`nvidia-smi`** on your base machine, you will also be able to run it in your Docker container (and all of your programs will be able to reference the GPU).

In order to use the NVIDIA Container Toolkit, simply pull the desired NVIDIA Container Toolkit image at the top of Dockerfile as the base image using command `FROM`. From this base state, you can develop the Dockerfile and add further layers as in normal Dockerfile.

## Example

```dockerfile
FROM nvidia/cuda:10.2-base
CMD nvidia-smi
```

In that Dockerfile we have imported the NVIDIA Container Toolkit image for 10.2 drivers and then we have specified a command to run when we run the container to check for the drivers.

Now we build the image like so with **`docker build . -t nvidia-test`**

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/image-41.png" title="Source: [How to Use the GPU within a Docker Container](https://blog.roboflow.com/use-the-gpu-in-docker/)" numbered="true" >}}

Now we run the container from the image by using the command **docker run --gpus all nvidia-test.** Keep in mind, we need the **`--gpus all`**,  otherwise  the GPU will not be exposed to the running container. For more details about specifying GPU(s), see [GPU Enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration).

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/image-42.png" title="Source: [How to Use the GPU within a Docker Container](https://blog.roboflow.com/use-the-gpu-in-docker/)" numbered="true" >}}



## Reference

- [How to Use the GPU within a Docker Container](https://blog.roboflow.com/use-the-gpu-in-docker/)

- [Nvidia container toolkit user guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html)