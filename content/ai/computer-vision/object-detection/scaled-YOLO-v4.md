---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 1009

# Basic metadata
title: "Scaled YOLOv4"
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
        weight: 9
---

Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao (more commonly known by their GitHub monikers, [WongKinYiu](https://github.com/WongKinYiu) and [AlexyAB](https://github.com/AlexeyAB)) have propelled the YOLOv4 model forward by efficiently scaling the network's design and scale, surpassing the previous state-of-the-art EfficientDet published earlier this year by the Google Research/Brain team.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/image.png" alt="img" style="zoom:80%;" />

## Train scaled YOLOv4 (PyTorch)

The Scaled-YOLOv4 implementation is written in the YOLOv5 PyTorch framework. Training scaled YOLOv4 is similar to [training YOLOv5]({{< relref "yolov5.md">}}). 

> Here is [t](https://github.com/WongKinYiu/ScaledYOLOv4/blob/yolov4-large/models/yolov4-csp.yaml)[he Scaled-YOLOv4 repo](https://github.com/WongKinYiu/ScaledYOLOv4/blob/yolov4-large/models/yolov4-csp.yaml), though you will notice that [WongKinYiu](https://github.com/WongKinYiu) has provided it there predominantly for research replication purposes and there are not many instructions for training on your own dataset. To train on your own data, our guide on [training YOLOv5 in PyTorch on custom data](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/) will be useful, as it is a very similar training procedure.

Tutorials from Roboflow:

- Video tutorial:

  {{< youtube rEbpKxZbvIo>}}

- [Blog post](https://blog.roboflow.com/how-to-train-scaled-yolov4/)

- [Google Colab Notebook](https://colab.research.google.com/drive/1LDmg0JRiC2N7_tx8wQoBzTB0jUZhywQr?usp=sharing)

My Colab Notebook: [yolov4_scaled.ipynb](https://colab.research.google.com/drive/1GfOzuMCpIcg1luILv7rehfY3Hk4p4SWc)

## Train scaled YOLOv4 (Darknet)

YOLOv4-csp training is also supported by [Darknet](https://github.com/AlexeyAB/darknet#pre-trained-models). Training yolov4-csp is similar to training yolov4 and yolov4-tiny. Slight difference:

- For config file, use [yolov4-csp.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-csp.cfg)
- For pretrained weights, use [yolov4-csp.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights)
- For pretrained convolutional layer weights, use [yolov4-csp.conv.142](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.conv.142)

## Reference

- Scaled YOLOv4 [paper](https://arxiv.org/abs/2011.08036)
- Github repo: [WongKinYiu](https://github.com/WongKinYiu)/**[ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)** (Different size of model in different branch)
- Blog post from AlexAB: [Scaled YOLO v4 is the best neural network for object detection on MS COCO dataset](https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982)

- Tutorials blog posts:
  - robolfow: [Scaled-YOLOv4 is Now the Best Model for Object Detection](https://blog.roboflow.com/scaled-yolov4-tops-efficientdet/)
  - [YOLOv4 团队最新力作！1774fps、COCO 最佳精度，分别适合高低端 GPU 的 YOLO](https://bbs.cvmart.net/articles/3674)
  - [上达最高精度，下到最快速度，Scaled-YOLOv4：模型缩放显神威](https://zhuanlan.zhihu.com/p/299385758)

