---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 115

# Basic metadata
title: "Body Pose"
date: 2021-07-07
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Lecture", "Pose Estimation"]
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
        parent: cv-lecture
        weight: 15
---

## Pose Recognition[^1]

![截屏2021-07-07 23.38.56](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-07%2023.38.56.png)

### **Speed** is the key

- Uses only one disparity image.

- It classifies each pixel independently.

- The process of feature extraction is *simultaneous* to the classification. 

- Simplest possible feature: difference between two pixels. 

- Classification is done through **Random Decision Forests**.

  - Learning
    - Randomly choose a set of thresholds and features for splits.
    - Pick the threshold and feature that provide the *largest* information gain. 
    - Recurse until a certain accuracy is reached or depth is obtained.

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-07%2023.41.56.png" alt="截屏2021-07-07 23.41.56" style="zoom:80%;" />

- Everything has an optimal GPU implementation.

### Training

- Key: using a huge amount of training data

- **Synthetic Training DB**

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-07%2023.45.46.png" alt="截屏2021-07-07 23.45.46" style="zoom: 67%;" />

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-07%2023.46.11.png" alt="截屏2021-07-07 23.46.11" style="zoom: 67%;" />

- Pixel classification results

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-07%2023.47.02.png" alt="截屏2021-07-07 23.47.02" style="zoom:67%;" />

### Joint estimation

- Use mean shift clustering on the pixels with **Gaussian kernel** to infer the center of clusters.
- Clustering is done in 3d space but every pixel is *weighted* by their world surface area to get depth invariance.
- Finally the *sum* of the weighted pixels is used as a **confidence measure**.

- Results

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-07%2023.49.21.png" alt="截屏2021-07-07 23.49.21" style="zoom:67%;" />

### Criticism 👎

- Not open source
- Biased towards upper body frontal poses
- Very difficult to improve or adapt.

## Pose Estimation without Kinect: Convolutional Pose Machines [^2]

### Pose Machine

- Unconstrained 2D-pose estimation on real world RGB images. 

- Outputs confidence maps for every joint of the skeleton. 

- Works in multiple stages refining the confidence maps.

- 💡 **Idea:**

  - **Local image evidence is weak (first stage confidence maps)**

  - **Part context can be a strong cue (confidence maps of other body joints)**

    **➔ Use confidence maps of all body joints of the previous stage to refine current results**

    ![截屏2021-07-08 00.02.49](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-08%2000.02.49.png)

#### Example

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-11%2012.38.10.png" alt="截屏2021-07-11 12.38.10" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-11%2012.38.27.png" alt="截屏2021-07-11 12.38.27" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-11%2012.38.40.png" alt="截屏2021-07-11 12.38.40" style="zoom:67%;" />



#### Details

We denote the pixel location of the $p$-th anatomical landmark (refer to as a **part**) $Y\_{p} \in \mathcal{Z} \subset \mathbb{R}^{2}$

- $\mathcal{Z}$: set of all $(u, v)$ locations in an image

🎯 **Goal: to predict the image locations $Y = (Y\_1, \dots, Y\_P)$ for all $P$ parts.**

![截屏2021-07-11 15.22.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-11%2015.22.48.png)

In each stage $t \in \{1 \dots T\}$, the classifiers $g\_t$ predict beliefs for assigning a location to each part $Y\_{p}=z, \forall z \in \mathcal{Z}$, based on 

- features extracted from the image at the location $z$ denoted by $\mathbf{x}\_{z} \in \mathbb{R}^{d}$ and
- contextual information from the preceding classifier in the neighborhood around each $Y\_p$ in stage $t$.

**First stage**

A classifier in the first stage $t = 1$ produces the following belief values:
$$
g\_1(\mathbf{x}\_z) \rightarrow \\{b\_1^p(Y\_p=z)\\}\_{p \in \\{0 \dots P\\}}
$$

- $b\_{1}^{p}\left(Y\_{p}=z\right)$: score predicted by the classifier $g\_1$ for assigning the $p$-th part in the first stage at image location $z$.

We represent all the beliefs of part $p$ evaluated at every location $z = (u, v)^T$ in the image as $\mathbf{b}\_{t}^{p} \in \mathbb{R}^{w \times h}$:
$$
\mathbf{b}\_{t}^{p}[u, v]=b\_{t}^{p}\left(Y\_{p}=z\right)
$$

- $w, h$: width and height of the image, respectively

For convenience, we denote the collection of belief maps for all the parts as $\mathbf{b}\_{t} \in \mathbb{R}^{w \times h \times(P+1)}$ ($+1$ for background)

**Subsequent stages**

The classifier predicts a belief for assigning a location to each part $Y\_{p}=z, \forall z \in \mathcal{Z}$,  based on

- features of the image data $\mathbf{x}\_{z}^{t} \in \mathbb{R}^{d}$ and
- contextual information from the preceeding classifier in the neighborhood around each $Y\_p$

$$
g\_{t}\left(\mathbf{x}\_{z}^{\prime}, \psi\_{t}\left(z, \mathbf{b}\_{t-1}\right)\right) \rightarrow \left \\{b\_{t}^{p}\left(Y\_{p}=z\right)\right\\}\_{p \in \\{0 \ldots P+1\\}}
$$

- $\psi\_{t>1}(\cdot)$: mapping from the beliefs $b\_{t−1}$ to context features.

In each stage, the computed beliefs provide an increasingly refined estimate for the location of each part.

### CPM

The prediction and image feature computation modules of a pose machine can be replaced by a deep convolutional architecture allowing for both image and contextual feature representations to be learned directly from data.

Advantage of convolutional architectures: completely differentiale $\rightarrow$ enabling end-to-end joint trainining of all stages 👍

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-08%2012.34.37.png" caption="CPM structure" numbered="true" >}}

### Learning in CPM

Potential problem of a network with a large number of layers: **vanishing gradient**

Solution

- Define a loss function at the output of each stage $t$ that minimizes the $l_2$ distance between the predicted ($b\_{t}^{p}$) and ideal ($b\_{*}^{p}\left(Y\_{p}=z\right)$) belief maps for each part. 

  - The ideal belief map for a part $p$, $b\_{*}^{p}\left(Y\_{p}=z\right)$, are created by putting Gaussian peaks at ground truth locations of each body part $p$.

  Cost function we aim to minimize at the output of each stage at each level:
  $$
  f\_{t}=\sum\_{p=1}^{P+1} \sum\_{z \in \mathcal{Z}}\left\|b\_{t}^{p}(z)-b\_{*}^{p}(z)\right\|\_{2}^{2} .
  $$

  - $P$: all body parts
  - $\mathcal{Z}$: set of all image locations in a believe map

- The overall objective for the full architecture is obtained by adding the losses at each stage:
  $$
  \mathcal{F}=\sum\_{t=1}^{T} f\_{t}
  $$
  

[^1]: J. Shotton et al., "Real-time human pose recognition in parts from single depth images," CVPR 2011, 2011, pp. 1297-1304, doi: 10.1109/CVPR.2011.5995316.
[^2]: Shih-En Wei, Varun Ramakrishna, Takeo Kanade, and Yaser Sheikh. Convolutional pose machines. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4724–4732, 2016

