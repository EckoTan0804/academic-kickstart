---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 1010

# Basic metadata
title: "YOLOv3: Train on Custom Dataset"
date: 2021-01-05
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Object Detection", "Scaled YOLOv4"]
categories: ["Computer Vision"]
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
    computer-vision:
        parent: object-detection
        weight: 10
---

Training YOLOv3 as well as YOLOv3 tiny on custom dataset is similar to [training YOLOv4 and YOLOv4 tiny]({{< relref "train-yolo-v4-custom-dataset.md" >}}). Only some steps need to be adjusted for YOLOv3 and YOLOv3 tiny:

- In step 1, we create our custom config file based on **cfg/yolov3.cfg** (YOLOv3) and **cfg/yolov3-tiny.cfg** (YOLOv3 tiny). Then adjust `batch`, `subdivisions`, `steps`, `width`, `height`, `classes`, and `filters` just as for YOLOv4.
- In step 6, download different pretrained weights for the convolutional layers
  - for `yolov3.cfg, yolov3-spp.cfg` (154 MB): [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
  - for `yolov3-tiny-prn.cfg , yolov3-tiny.cfg` (6 MB): [yolov3-tiny.conv.11](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)

## Reference

- Tutorial from darknet repo: [How to train (to detect your custom objects)](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

- [How to train YOLOv3 on the custom dataset](https://thebinarynotes.com/how-to-train-yolov3-custom-dataset/)