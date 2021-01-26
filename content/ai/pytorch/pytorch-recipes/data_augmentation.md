---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 907

# Basic metadata
title: "Data Augmentation"
date: 2021-01-17
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "PyTorch Recipe", "Data Augmentation"]
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
        parent: pytorch-recipes
        weight: 7
---

## What is data augmentation?

To solve the problem that it's hard to get enough data for training neural networks, **image augmentation is a process of creating new training examples from the existing ones. To make a new sample, you slightly change the original image.**

For instance, you could make a new image a little brighter; you could cut a piece from the original image; you could make a new image by mirroring the original one, etc. Here are some examples of transformations of the original image that will create a new training sample:

![augmentation](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/augmentation.jpg)

By applying those transformations to the original training dataset, you could create an almost infinite amount of new training samples.

## Premise of data augmentation

A [convolutional neural network](https://nanonets.com/blog/human-pose-estimation-2d-guide/) that can robustly classify objects even if its placed in different orientations is said to have the property called **invariance**. More specifically, a CNN can be invariant to **translation, viewpoint, size** or **illumination** (Or a combination of the above).

## When to apply augmentation?

The answer may seem quite obvious; we do augmentation **before** we feed the data to the model.

However, we have two options here:

- **Offline augmentation**
  - Preferred for relatively **smaller datasets**
  - Increasing the size of the dataset by a factor equal to the number of transformations we perform
    - For example, by **flipping** all my images, I would **increase the size** of my odataset by a **factor of 2**
- **Online augmentation / Augmentation on the fly**
  - Preferred for **larger datasets**, as we can’t afford the explosive increase in size.
  - Perform transformations **on the mini-batches** that we would feed to our model. 

## Use data augmentation in the right way

‼️ **Do NOT increase irrelevant data!!!**

Sometimes not all augmentation techniques make sense for a dataset. Consider the following car example:

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*vW3KGPp_w0wN6k3gYVlVHA.jpeg" title="The first image (from the left) is the original, the second one is flipped horizontally, the third one is rotated by 180 degrees, and the last one is rotated by 90 degrees (clockwise)." numbered="true" >}}

They are pictures of the same car, but our target application may NEVER see cars presented in these orientations. For example, if we're gonna classify random cars on the road, only the second image would make sense to be in the dataset.

## How to conduct data augmentation in PyTorch?

### Use `torchvision.transforms`

- Provides common image transformations
- Can be chained together using `transforms.Compose`

### 🔥 Use [`albumentations`](https://github.com/albumentations-team/albumentations)



#### Demo

[Demo](https://albumentations-demo.herokuapp.com/) for viewing different augmentation transformations



## Reference

- [Data Augmentation | How to use Deep Learning when you have Limited Data — Part 2](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/)

