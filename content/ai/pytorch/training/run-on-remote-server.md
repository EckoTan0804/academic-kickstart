---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 1002

# Basic metadata
title: "Running Jupyter Notebook/Lab on a remote server"
date: 2020-11-29
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "NN Training"]
categories: ["Deep Learning"]
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
    pytorch:
        parent: training
        weight: 2
---

## **SSH Tunneling**

- Also known as **port forwarding**. I.e. setup a network tunnel (a connection for data to flow) from a local point to remote point.

- Example:

  ```bash
  ssh username@xx.xx.xx.xx -NL 1234:localhost:1234
  ```

  `1234:localhost:1234` means that any network request you send to port `1234` in your current system will be automatically forwarded to `localhost:1234` *from the remote system*.

  ![1*uGLPZIeLPkvvaRkVG1-tkw](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*uGLPZIeLPkvvaRkVG1-tkw.png)

## Run Jupyter Notebook on a remote server

### 0. Log on the remote server vis SSH

### 1. Run Jupyter Notebook from remote machine

To launch Jupyter Notebook from remote server, type the following in the **remote** server console:

```bash
jupyter notebook --no-browser --port=<remote_port>
```

- `--no-browser`: this starts the notebook without opening a browser
- `--port=<remote_port>`: this sets the port for starting your notebook where the default is `8888`. When it’s occupied, it finds the next available port.

Note: Please note the port setting. You will need it in the next step.

### 2. Forward `<remote_port>` to `<local_port>` and listen to it

In your remote, the notebook is now running at the port ``<remote_port>`` that you specified. What you’ll do next is forward this to port `<local_port>` *of your **local** machine* via [SSH tunneling](#ssh-tunnel) so that you can listen and run it from your browser.

Type the following in the **local** machine terminal:

```bash
ssh -N -L <local_port>:localhost:<remote_port> <remote_user>@<remote_host>
```

This opens up a new SSH session in the terminal.

### 3. Fire-up Jupyter Notebook

To open up the Jupyter notebook from your remote machine, simply start your browser and navigate to `localhost:<loacl_port>`

### Overview

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/jupyternotebook.png" alt="overview" style="zoom: 50%;" />



## Run Jupyter Lab on a remote server

It's similar to running Jupyter Notebook. Simply replace `notebook` with `lab`.

## Reference

- Jupyter Notebook
  - :thumbsup: [Running a Jupyter notebook from a remote server](https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/)
  - [Running Jupyter Notebook on a remote server](https://docs.anaconda.com/anaconda/user-guide/tasks/remote-jupyter-notebook/)

- Jupyter Lab
  - [Remote Jupyter Lab: how to utilize Jupyter Lab to its fullest on a remote server?](https://medium.com/spencerweekly/remote-jupyter-lab-how-to-utilize-jupyter-lab-to-its-fullest-on-a-remote-server-2a359159d2f6)

