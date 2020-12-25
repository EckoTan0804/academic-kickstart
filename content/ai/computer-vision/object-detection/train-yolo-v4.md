---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 1003

# Basic metadata
title: "YOLOv4: Run Pretrained YOLOv4 on COCO Dataset"
date: 2020-11-04
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Object Detection", "YOLOv4"]
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
        weight: 3
---

Here we will learn how to get YOLOv4 Object Detection running in the Cloud with Google Colab step by step.

Check out the [Google Colab Notebook](https://colab.research.google.com/drive/1o-xfVm7A-kgtFZRrehJvnibuBwzNPs1-?authuser=1#scrollTo=P5WqSvgwqmLT)

## Clone and build DarkNet

Clone darknet from AlexeyAB's [repository](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects), 

```bash
!git clone https://github.com/AlexeyAB/darknet
```

Adjust the Makefile to enable OPENCV and GPU for darknet

```bash
# change makefile to have GPU and OPENCV enabled
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```

Verify CUDA

```bash
# verify CUDA
!/usr/local/cuda/bin/nvcc --version
```

Build darknet

> Note: Do not worry about any warnings when running the `!make` cell!

```bash
# make darknet 
# (builds darknet so that you can then use the darknet executable file 
# to run or train object detectors)
!make
```

## Download pretrained YOLO v4 weights

YOLOv4 has been trained already on the coco dataset which has 80 classes that it can predict. We will grab these pretrained weights so that we can run YOLOv4 on these pretrained classes and get detections.

```bash
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

## Define helper functions

```python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline


def imShow(path):
    """
    Show image
    """
    image = cv2.imread(path)
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (3*width, 3*height), interpolation = cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()


def upload():
    """
    upload files to Google Colab
    """
    from google.colab import files
    uploaded = files.upload()
    for name, data in uploaded.items():
        with open(name, 'wb') as f:
            f.write(data)
            print(f'saved file {name}')


def download(path):
    """
    Download from Google Colab
    """
    from google.colab import files
    files.download(path)
```

## Run detections with Darknet and YOLOv4

The object detector can be run using the following command

```bash
!./darknet detector test <path to .data file> <path to config> <path to weights> <path to image>
```

This will output the image with the detections shown. The most recent detections are always saved to '**predictions.jpg**'

**Note:** After running detections OpenCV can't open the image instantly in the cloud so we must run:

```python
imShow('predictions.jpg')
```

Darknet comes with a few images already installed in the `darknet/data/` folder. Let's test one of the images inside:

```bash
# run darknet detection on test images
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/person.jpg
```

```python
imShow('predictions.jpg')
```

![predictions](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/predictions.png)

### Run detections using uploaded image

We can also mount Google drive into the cloud VM a

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

```bash
# this creates a symbolic link 
# so that now the path /content/gdrive/My\ Drive/ is equal to /mydrive
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive
```

nd run YOLOv4 with images from Google drive using the following command:

```bash
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights /mydrive/<path to image>
```

For example, I uploaded an image called "pedestrian.jpg" in `images/` folder:

![pedestrian](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/pedestrian.jpg)

and run detection on it:

```python
!./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights /mydrive/images/pedestrian.jpg
imShow('predictions.jpg')
```

![pedestrian_predictions](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/pedestrian_predictions.png)

## Reference

- YOLOv4 in the CLOUD: Install and Run Object Detector (FREE GPU)

  - [Google Colab Notebook](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing#scrollTo=iZULaGX7_H1u)

  - https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial

  - Video Tutorial

    {{< youtube mKAEGSxwOAY >}}

