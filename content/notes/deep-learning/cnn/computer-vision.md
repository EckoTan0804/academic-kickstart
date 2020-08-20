---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 440

# Basic metadata
title: "Computer Vision"
date: 2020-08-20
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Optimization", "CNN"]
categories: ["Deep Learning"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?

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
    deep-learning:
        parent: cnn
        weight: 4

---

## Computer Vision (CV) Tasks

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2013.23.32.png" alt="截屏2020-08-20 13.23.32" style="zoom: 50%;" />

- Classification
- Classification + [Localization](#object-localization-:-coordinate-prediction)
- [Object Detection](#detection)
- [Instance Segmentation](#instance-segmentation)

## **Object Localization: Coordinate prediction**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2013.36.50.png" alt="截屏2020-08-20 13.36.50" style="zoom:67%;" />

### Sliding Window

**Object Localization**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2016.52.34.png" alt="截屏2020-08-20 16.52.34" style="zoom:67%;" />

**Classification & Localization**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2016.53.21.png" alt="截屏2020-08-20 16.53.21" style="zoom:67%;" />

## Detection

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2016.54.26.png" alt="截屏2020-08-20 16.54.26" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2016.54.44.png" alt="截屏2020-08-20 16.54.44" style="zoom:50%;" />

Sliding Window + Classification:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2016.55.40.png" alt="截屏2020-08-20 16.55.40" style="zoom:67%;" />

### Regioning

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2016.56.37.png" alt="截屏2020-08-20 16.56.37" style="zoom:67%;" />

- <span style="color:Red">Sliding Window Problem: Need to test many positions and scales, and use a computationally demanding classifier</span>

- Solution: Only look at a tiny subset of possible positions
  - Regioning => propose image regions that are likely to contain objects 
  - Classify individual regions and correct regions
  - R-CNN -> Fast R-CNN -> Faster R-CNN

#### R-CNN

- Propose approx. 2k different regions (bounding boxes) for image classification
- For each box, do image classification with CNN 
  - Discard unlikely boxes
-  Refine bounding boxes with regression

![Object Detection for Dummies Part 3: R-CNN Family](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/RCNN.png)

#### Fast R-CNN

- 9x faster training, 213x faster test time
- R-CNN is not end to end (first train softmax classifier, use that for training bounding box regressor)
- Similar to R-CNN
  - Apply Region Proposals on feature map result of applied CNN to input image
  - Reshape region proposals on feature map into fixed size 
  - Feed into FC layer

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/fast-RCNN.png" alt="Object Detection for Dummies Part 3: R-CNN Family" style="zoom: 50%;" />

#### Faster R-CNN

- Both R-CNN and R-CNN rely on Selective Search for region proposals -> most time consuming part 🤪
- Use a seperate Network for predicting the regions of interest :muscle:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/faster-RCNN.png" alt="Object Detection for Dummies Part 3: R-CNN Family" style="zoom: 67%;" />

### YOLO

- **Y**ou **O**nly **L**ook **O**nce: Unified Real-Time Object Detection

- „Simple network“, directly from pixels to bounding box / object detection / class prediction

  

## Image Segmentation

- Grouping Pixels into regions that belong to same properties
- Eg: Segmenting an Image into meaningful objects

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2017.31.10.png" alt="截屏2020-08-20 17.31.10" style="zoom:67%;" />

### Semantic Segmentation

**Sliding Window**

- Label each pixel in image with a category label

- Don‘t differentiate instances, only care about pixels

- => just extract small patches from an image and classify center pixel with a normal CNN classifier

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2017.33.57.png" alt="截屏2020-08-20 17.33.57" style="zoom:67%;" />

- <span style="color:Red">Problem: very inefficient</span>

**Fully convolutional**

- Keep the network as an end to end convolutional Neural Network

- Predictions are made for all pixels at once

  ![截屏2020-08-20 17.34.17](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-20%2017.34.17.png)

- Convolutions at original image resolution are very expensive

## Reference

- [Object Detection for Dummies Part 3: R-CNN Family](https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html)









