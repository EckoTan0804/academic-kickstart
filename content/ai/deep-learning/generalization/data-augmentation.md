---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 153

# Basic metadata
title: "👍 Data Augmentation"
date: 2020-08-16
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Generalization"]
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
        parent: generalization
        weight: 3

---

## Motivation

Overfitting happens because of having **too few examples** to train on, resulting in a model that has poor generalization performance :cry:. If we had infinite training data, we wouldn’t overfit because we would see every possible instance.

However, in most machine learning applications, especially in image classification tasks,  obtaining new training data is not easy. Therefore we need to make do with the training set at hand. :muscle:

**Data augmentation is a way to generate more training data from our current set. It enriches or “augments” the training data by generating new examples via random transformation of existing ones. This way we artificially boost the size of the training set, reducing overfitting. So data augmentation can also be considered as a regularization technique.**

Data augmentation is done dynamically during training time. We need to generate realistic images, and the transformations should be learnable, simply adding noise won’t help. Common transformations are

- rotation
- shifting
- resizing
- exposure adjustment
- contrast change 
- etc.

This way we can generate a lot of new samples from a single training example. 

**Notice that data augmentation is ONLY performed on the training data, we don’t touch the validation or test set.**

## Popular Augmentation Techniques

### Flip

{{< figure src="https://nanonets.com/blog/content/images/2018/11/1_-beH1nNqlm_Wj-0PcWUKTw.jpeg" title="Left: original image. Middle: image flipped horizontally. Right: image flipped vertically" numbered="true" >}}

### Rotation

{{< figure src="https://cdn-images-1.medium.com/max/720/1*i_F6aNKj3yggkcNXQxYA4A.jpeg" title="Example of square images rotated at right angles. From left to right: The images are rotated by 90 degrees clockwise with respect to the previous one." numbered="true" >}}

Note: image dimensions may not be preserved after rotation

- If image is a square, rotating it at right angles will preserve the image size. 
- If image is a rectangle, rotating it by 180 degrees would preserve the size.

### Scale

{{< figure src="https://cdn-images-1.medium.com/max/720/1*INLTn7GWM-m69GUwFzPOaQ.jpeg" title="Left: original image. Middle: image scaled outward by 10%. Right: image scaled outward by 20%" numbered="true" >}}

The image can be scaled outward or inward. While scaling outward, the final image size will be larger than the original image size. Most image frameworks cut out a section from the new image, with size equal to the original image. 

### Crop

{{< figure src="https://cdn-images-1.medium.com/max/720/1*ypuimiaLtg_9KaQwltrxJQ.jpeg" title="Left: original image. Middle: a square section cropped from the top-left. Right: a square section cropped from the bottom-right. The cropped sections were resized to the original image size." numbered="true" >}}

Random cropping

1. Randomly sample a section from the original image
2. Resize this section to the original image size

### Translation

{{< figure src="https://cdn-images-1.medium.com/max/720/1*L07HTRw7zuHGT4oYEMlDig.jpeg" title="Left: original image. Middle: the image translated to the right. Right: the image translated upwards." numbered="true" >}}

**Translation = moving the image along the X or Y direction (or both)**

This method of augmentation is very useful as most objects can be located at almost anywhere in the image. This forces your convolutional neural network to look everywhere.

### Gaussian Noise

{{< figure src="https://cdn-images-1.medium.com/max/720/1*cx24OpSNOwgg7ULUHKiGnA.png" title="Left: original image. Middle: image with added Gaussian noise. Right: image with added salt and pepper noise." numbered="true" >}}

One reason of overfitting ist that neural network tries to learn high frequency features (patterns that occur a lot) that may not be useful.

**Gaussian noise**, which has zero mean, essentially has data points in all frequencies, effectively distorting the high frequency features. This also means that lower frequency components (usually, your intended data) are also distorted, but your neural network can learn to look past that. Adding just the right amount of noise can enhance the learning capability.

A toned down version of this is the **salt and pepper noise**, which presents itself as random black and white pixels spread through the image. This is similar to the effect produced by adding Gaussian noise to an image, but may have a lower information distortion level.

## Reference

- [Applied Deep Learning - Part 4: Convolutional Neural Networks](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2#9722)
- [Data Augmentation | How to use Deep Learning when you have Limited Data — Part 2](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/)