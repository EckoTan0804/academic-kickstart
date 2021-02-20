---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 20

# Basic metadata
title: "People Detection: Global Approaches"
date: 2021-02-18
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Lecture", "Person Detection"]
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
        weight: 10
---

## Motivation

### Why people detection?

- Person Re-Identification 
- Person Tracking

- Security (e.g. Border Control) 
- Automotive (e.g. Collision Prevention)
- Interaction (e.g. Xbox Kinect)

- Medical (e.g. Patient Monitoring) 
- Commercial (e.g. Customer Counting)

### Why is people detection difficult?

- **Clothing**

  Large variety of clothing styles causes greater appearance variety

- **Accessories**
  Occlusions by accessories. E.g. backpack, umbrella, handbag, ...

- **Articulation**
  Faces are mostly rigid. Persons can take on many different poses

- **Clutter**
  People frequently overlap each other in images (crowds)

## Categories

### Still image vs. video

**Still image based** 

- Mostly based on gray-value information from visual images
- Other possible cues: color, infra-red, radar, stereo
- 👍 Advantage: Applicable in wider variety of applications
- 👎 Disadvantages
  - Often more difficult (only a single frame)
  - Performs poorer than video based techniques

**Video based**

- Background modeling
- Temporal information (speed, position in earlier frames)
- Optical flow
- Can be (re-)initialized by still image approach

- 👎 Disadvantage: Hard to apply in unconstrained scenarios

### Global vs. parts

**Global approaches**

- Holistic model, e.g. one feature for whole person

  ![截屏2021-02-19 23.32.50](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2023.32.50.png)

- 👍 Advantages
  - typically simple model
  - work well for low resolutions
- 👎 Disadvantages
  - problems with occlusions 
  - problems with articulations

**Part-based approaches**

- Model body sub-parts separately

  ![截屏2021-02-19 23.33.55](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2023.33.55.png)

- 👍 Advantages
  - deal better with moving body parts (poses)
  - able to handle occlusions, overlaps
  - sharing of training data
- 👎 Disadvantages
  - require more complex reasoning 
  - problems with low resolutions

### discriminative vs. generative

**Generative model**

- Models how data (i.e. person images) is generated
- 👍 Advantages
  - possibly interpretable, i.e. know why reject/accept
  - models the object class/can draw samples

- 👎 Disadvantages
  - model variability unimportant to classification task 
  - often hard to build good model with few parameters

**Discriminative model**

- Can only discriminate for given data, if it is a person or not
- 👍 Advantages
  - appealing when infeasible to model data itself
  - currently often excel in practice

- 👎 Disadvantages
  - often can’t provide uncertainty in predictions
  - non-interpretable

## Typical components of global approaches

### Detection via classification (binary classifier)

**Sliding window**: Scan window at different **positions and scales**

![截屏2021-02-20 13.49.47](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2013.49.47.png)

### Gradient based

- Popular and successful in the vision community

- Avoid hard decisions (compared to edge based features)

- Examples

  - **Histogram of Oriented Gradients (HOG)**
  - **Scale-Invariant Feature Transform (SIFT)**
  - **Gradient Location and Orientation Histogram (GLOH)**

- Computing gradients

  - Centered
    $$
    f^{\prime}(x)=\lim \_{h \rightarrow 0} \frac{f(x+h)-f(x-h)}{2 h}
    $$

  - Gradient **magnitude**
    $$
    s = \sqrt{s\_x^2 + s\_y^2}
    $$

  - Gradient **orientation**
    $$
    \theta=\arctan \left(\frac{s\_{y}}{s\_{x}}\right)
    $$
    ![截屏2021-02-20 13.55.54](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2013.55.54.png)

- Gradient in image

  - Image: discrete, 2-dimensional signal

  - Use filter mask to compute gradient

    - $x$-direction:

      <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2014.50.04.png" alt="截屏2021-02-20 14.50.04" style="zoom: 67%;" />

    - $y$-direction

      <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2014.50.15.png" alt="截屏2021-02-20 14.50.15" style="zoom:67%;" />





### Edge based

### Wavelet based

## HOG people detector [^1]

- Gradient-based feature descriptor developed for people detection 
- Global descriptor for the complete body
- High-dimensional (typically ~4000 dimensions)

- Very promising results on challenging data sets

### Phases

#### Learning Phase

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.13.56.png" alt="截屏2021-02-20 17.13.56" style="zoom:80%;" />

#### Detection Phase

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.14.25.png" alt="截屏2021-02-20 17.14.25" style="zoom:80%;" />

### How HOG descriptor works?

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.19.38.png" alt="截屏2021-02-20 17.19.38" style="zoom:80%;" />

1. Compute gradients on an image region of 64x128 pixels
2. Compute gradient orientation histograms on *cells* of 8x8 pixels (in total 8x16 cells).
    typical histogram size: 9 bins
3. Normalize histograms within overlapping *blocks* of 2x2 cells (in total 7x15 blocks)
    block descriptor size: 4x9 = 36
4. Concatenate block descriptors $\rightarrow$ 7 x 15 x 4 x 9 = 3780 dimensional feature vector

#### 1. Gradients

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.21.44.png" alt="截屏2021-02-20 17.21.44" style="zoom:80%;" />

- Convolution with [-1 0 1] filters (x and y direction)

- Compute gradient magnitude and direction

  - Per pixel: color channel with greatest magnitude is used for final gradient (color is used!)

    ![截屏2021-02-20 17.22.45](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.22.45.png)

#### 2. **Cell histograms**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.23.45.png" alt="截屏2021-02-20 17.23.45" style="zoom:67%;" />

- 9 bins for gradient orientations (0-180 degrees)
- Filled with magnitudes
- Interpolated trilinearly
  - bilinearly into spatial cells 
  - linearly into orientation bins

#### 3. Blocks

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.25.24.png" alt="截屏2021-02-20 17.25.24" style="zoom:80%;" />

- Overlapping blocks of 2x2 cells
- Cell histograms are concatenated and then normalized
- Normalization
  - different norms possible (L2, L2hys etc.)
  - add a normalization epsilon to avoid division by zero

#### 4. **The final HOG descriptor**

Concatenation of block descriptors

![截屏2021-02-20 17.32.07](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.32.07.png)

Visualization

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.32.24.png" alt="截屏2021-02-20 17.32.24" style="zoom:80%;" />

### From feature to detector

- Simple linear SVM on top of the HOG Features

  - Fast (one inner product per evaluation window)

    for an entire image it’s a vector-matrix multiplication

- Gaussian kernel SVM

  - slightly better classification accuracy

  - but considerable increase in computation time



## Silhouette Matching [^2]

### Idea

- 🎯 **Goal: align known object shapes with image**

  ![截屏2021-02-20 17.38.46](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.38.46.png)

- Requirements for an alignment algorithm
  - high detection rate
  - few false positives 
  - robustness

  - computationally inexpensive

### Computational complexity

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.41.11.png" alt="截屏2021-02-20 17.41.11" style="zoom:67%;" />

Complexity is **O(#positions * #templates * #contourpixels * sizeof(searchregion))**

### Distance transform

Used to compare/align two (typically binary) shapes

![截屏2021-02-20 17.44.59](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.44.59.png)

1. Compute the distance from each pixel to the nearest edge pixel

   - here the euclidean distances are approximated by the **2-3 distance**

   <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.45.23.png" alt="截屏2021-02-20 17.45.23" style="zoom:80%;" />

2. Overlay second shape over distance transform

   ![截屏2021-02-20 17.45.42](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.45.42.png)

3. Accumulate distances along shape 2

4. Find best matching position by an exhaustive search

However:

- 2-3 distance is not symmetric
- 2-3 distance has to be normalized w.r.t. the length of the shapes

### Chamfer matching

![截屏2021-02-20 17.47.50](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.47.50.png)

#### **Efficient Implementation**

The distance transform can be efficiently computed by two scans over the complete image

- **Forward-Scan**

  - starts in the upper-left corner and moves from left to right, top to bottom

  - uses the following mask

    ![截屏2021-02-20 17.50.24](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.50.24.png)

- **Backward-Scan**

  - starts in the lower-right corner and moves from right to left, bottom to top

  - uses the following mask

    ![截屏2021-02-20 17.50.50](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.50.50.png)

Advantages

- Fast
- Good performance on uncluttered images (with few background structures)

Disadvantages

- Bad performance for cluttered images
- Needs a huge number of people silhouettes

### Template Hierarchy

- Reduce the number of silhouettes to consider
- The shapes are clustered by similarity

![截屏2021-02-20 17.52.49](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-20%2017.52.49.png)

### Coarse-To-Fine Search

- Goal: Reduce search effort by discarding unlikely regions with minimal computational effort

- Idea:

  - subsample the image and search first at a coarse scale

  - only consider regions with a low distance when searching for a match on finer scales

- Need to find reasonable thresholds





[^1]: N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 886-893 vol. 1, doi: 10.1109/CVPR.2005.177.
[^2]: D. M. Gavrila and V. Philomin, "Real-time object detection for "smart" vehicles," Proceedings of the Seventh IEEE International Conference on Computer Vision, Kerkyra, Greece, 1999, pp. 87-93 vol.1, doi: 10.1109/ICCV.1999.791202.