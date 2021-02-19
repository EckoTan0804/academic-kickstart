---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 17

# Basic metadata
title: "Facial Feature Detection"
date: 2021-02-18
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Lecture", "Face"]
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
        weight: 8
---

## Introduction

### What are facial features?

Facial features are referred to as **salient parts of a face region which carry meaningful information**.

- E.g. eye, eyeblow, nose, mouth
- A.k.a {{< hl >}}**facial landmarks**{{< /hl >}}

### What is facial feature detection?

Facial feature detection is defined as methods of **locating the specific areas of a face**.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-18%2023.13.42.png" alt="截屏2021-02-18 23.13.42" style="zoom:80%;" />

### Applications of facial feature detection

- Face recognition
- Model-based head pose estimation
- Eye gaze tracking
- Facial expression recognition
- Age modeling

### Problems in facial feature detection

- **Identity variations**

  Each person has unique facial part

- **Expression variations**

  Some facial features change their state (e.g. eye blinks).

- **Head rotations**

  If a head orientation changes, the visual appearance also changes.

- **Scale variations**

  Changes in resolution and distance to the camera affect appearance.

- **Lighting conditions**

  Light has non-linear effects on the pixel values of a image.

- **Occlusions**

  Hair or glasses might hide facial features.

## Older approaches (from face detection)

- Integral projections + geometric constraints
- Haar-Filter Cascades
- PCA-based methods (Modular Eigenspace)
- Morphable 3D Model

## Statistical appearance models

- 💡 Idea: make use of prior-knowledge, i.e. models, to reduce the complexity of the task
- Needs to be able to deal with variability $\rightarrow$ **deformable models**
- Use statistical models of shape and texture to find facial landmark points
- Good models should
  - Capture the various characteristics of the object to be detected 
  - Be a compact representation in order to avoid heavy calculation 
  - Be robust against noise

### Basic idea

1. **Training** stage: construction of models
2. **Test** stage: Search the region of interest (ROI)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2011.30.55.png" alt="截屏2021-02-19 11.30.55" style="zoom:80%;" />

### Appearance models

- Represent both **texture** and **shape** 

- Statistical model learned from training data

- Modeling shape variability

  - Landmark points

  $$
  x=\left[x\_{1}, y\_{1}, x\_{2}, y\_{2}, \ldots, x\_{n}, y\_{n}\right]^{T}
  $$

  - Model
    $$
    x \approx \bar{x}+P\_{s} b\_{s}
    $$

    - $\bar{x}$: Mean vector
    - $P\_s$: Eigenvectors of covariance matrix
    - $b\_s = P\_s^T(x - \bar{x})$



- Modeling intensity variability:

  - Gray values
    $$
    h=\left[g\_{1}, g\_{2}, \ldots, g\_{k}\right]^{T}
    $$

  - Model
    $$
    h \approx \bar{h} + P\_ib\_i
    $$

    - $\bar{h}$: Mean vector
    - $P\_s$: Eigenvectors of covariance matrix
    - $b\_i = P\_i^T(h - \bar{h})$

### Training of appearance models

#### 1. Construct a shape model with Principal component analysis (PCA)

A shape is represented with manually labeled points. 

![截屏2021-02-19 11.40.18](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2011.40.18.png)

The shape model approximates the shape of an object.

##### **Procrustes Analysis**

Align the shapes all together to remove translation, rotation and scaling

![截屏2021-02-19 11.51.46](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2011.51.46.png)

![截屏2021-02-19 11.52.05](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2011.52.05.png)

**PCA**

The positions of labeled points are
$$
x = \bar{x}+P\_{s} b\_{s}
$$

- $\bar{x}$: Mean shape
- $P\_s$: Orthogonal modes of variation obtained by PCA
- $b\_s$: Shape parameters in the projected space

The shapes are represented with fewer parameters ($\operatorname{Dim}(x) > \operatorname{Dim}(b\_s)$)

Generating plausible shapes:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2012.38.11.png" alt="截屏2021-02-19 12.38.11" style="zoom:80%;" />

#### 2. Construct a texture model which represents grey-scale (or color) values at each point

Warp the image so that the labeled points fit on the mean shape

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2012.39.55.png" alt="截屏2021-02-19 12.39.55" style="zoom:80%;" />

Then normalize the intensity on the *shape-free* patch.

##### Texture warping

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2012.58.42.png" alt="截屏2021-02-19 12.58.42"  />

#### Texture model

The pixel values on the shape-free patch
$$
g = \bar{g} + P\_g b\_g
$$

- $\bar{g}$ : Mean of normalized pixel values
- $P\_g$ : Orthogonal modes of variation obtained by PCA
- $b\_g$: Texture parameters in the projected space

The pixel values (appearance) are presented with fewer parameters ($\operatorname{Dim}(g) > \operatorname{Dim}(b\_g)$)

#### 3. Model the correlation between shapes and grey-level models

The concatenated vector is
$$
b=\left(\begin{array}{c}
W\_{s} b\_{s} \\\\
b\_{g}
\end{array}\right)
$$
Apply PCA:
$$
b=P\_{c} c=\left(\begin{array}{l}
P\_{c s} \\\\
P\_{c g}
\end{array}\right)c
$$
Now the parameter $\mathbf{c}$ can control both shape and grey-level models

- The shape model
  $$
  x=\bar{x}+P\_{s} W\_{s}^{-1} P\_{c s} c
  $$

- The grey-level model
  $$
  g=\bar{g}+P\_{g} P\_{c g} c
  $$

**Examples of synthesized faces**

Various objects can be synthesized by controlling the parameter $\mathbf{c}$

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2013.07.53.png" alt="截屏2021-02-19 13.07.53" style="zoom:80%;" />

### Dataset for Building Model

IMM data set from Danish Technical University

- 240 images with 640*480 size; 40 individuals, with 36 males and 4 females.

- Each Subject 6 shots, with different pose, expressions and illuminations.

- Each image is labeled with 58 landmarks; 3 closed and 4 opened point-paths.

  ![截屏2021-02-19 13.09.03](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2013.09.03.png)

### Image Interpretation with Models

- 🎯 **Goal: find the set of parameters which best match the model to the image**
  - Optimize some cost function
  - Difficult optimization problem
- Set of parameters 
  - Defines shape, position, appearance
  - Can be used for further processing
    - Position of landmarks

    - Face recognition

    - Facial expression recognition 
    - Pose estimation
- Problem: Optimizing the model fit
  - [Active Shape Models](#active-shape-models-asm)
  - [Active Appearance Models](#active-appearance-models-aam)

### Active Shape Models (ASM)

Given a rough starting position, create an instance of $\mathbf{X}$ of the model using 

- shape parameters $b$
- translation $T=(X\_t,Y\_t)$
- scale $s$ 
- rotation $\theta$

Iterative approach:

1. Examine region of the image around $\mathbf{X}\_i$ to find the best nearby match for the point $\mathbf{X}\_i^\prime$
2. Update parameters $(b, T, s, \theta)$ to best fit the new points $\mathbf{X}$ (constrain the model parameters to be within three standard deviations)
3. Repeat until convergence

In practice: **search along profile normals** 

- The optimal parameters are searched from **multi-resolution** images hierarchically (faster algorithm)

  1. Search for the object in a coarse image
  2. Refine the location in a series of higher resolution images.

  ![截屏2021-02-19 13.31.48](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2013.31.48.png)

**Example of search**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2013.32.21.png" alt="截屏2021-02-19 13.32.21" style="zoom:80%;" />



#### Disadvantages

- Uses mainly shape constraints for search

- Does not take advantage of texture across the target

### Active Appearance Models (AAM)

- Optimize parameters, so as to minimize the difference of a synthesized image and the target image
- Solved using a gradient-descent approach

#### Fitting AAMs

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2015.51.19.png" alt="截屏2021-02-19 15.51.19" style="zoom:80%;" />

Learning linear relation matrix $\mathbf{R}$ using multi-variate linear regression

- Generate training set by perturbing model parameters for training images 
- Include small displacements in position, scale, and orientation

- Record perturbation and image difference

- Experimentally, optimal perturbation around 0.5 standard deviations for each parameter

### ASM vs. AAM

**ASM**

- Seeks to match a set of model points to an image, constrained by a statistical model of shape
- Matches model points using an **iterative** technique (variant of EM-algorithm)
- A search is made around the current position of each point to find a nearby point which best matches texture for the landmark
- Parameters of the shape model are then updated to move the model points closer to the new points in the image

**AAM**: matches both position of model points and representation of texture of the object to an image

- Uses the difference between current synthesized image and target image to update parameters
- Typically, less landmark points are needed

### Summary of ASM and AAM

- Statistical appearance models provide a compact representation
- Can model variations such as different identities, facial expression, appearances, etc.
- Labeled training images are needed (very time-consuming) 🤪
- Original formulation of ASM and AAM is computationally expensive (i.e. slow) 🤪

- But, efficient extensions and speed-ups exist!
  - Multi-resolution search
  - Constrained AAM search
  - Inverse compositional AAMs (CMU)

- Usage
  - **Facial fiducial point detection**
  - Face recognition, pose estimation 
  - Facial expression analysis 
  - Audio-visual speech recognition

## More Modern Approaches: **Conditional Random Forests** For Real Time Facial Feature Detection[^1]

### Basics

#### Regression tree

- Basically like classification decision tree

- In the nodes-decisions are comparison of numbers

- In the leafs-numbers or multidimensional vectors of numbers

- Example

  ![截屏2021-02-19 16.02.51](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2016.02.51.png)

#### Random regression forests

- Set of random regression trees

- Random

  - Different trees trained on random subset of training data

  - After training, predictions for unseen samples can be made by averaging the predictions from all the individual regression trees

    ![截屏2021-02-19 16.03.52](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2016.03.52.png)

### Basic idea

- Train different set of trees for different head pose.

- The leaf nodes accumulates votes for the different facial fiducial points

![截屏2021-02-19 16.21.08](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2016.21.08.png)

### Regression forests training

- Each Tree is trained from randomly selected set of images.
- Extract patches in each image

- Training goal: accumulate probability for a feature point $C\_n$ given a patch $P$ at the leaf node
  - Each patch is represented by appearance features $I$, and displacement vectors $D$ (offsets) to each of the facial fiducial feature point. I.e. $P = \\{I, D\\}$
  - A simple patch comparison is used as Tree-node splitting criterion

###Regression forests testing

- Given: a random face image
- Extract densely set of patches from the image
- Feed all patches to all trees in the forest
- Get for each patch $P\_i$ a corresponding set of leafs
- A density estimator for the location of ffp's is calculated
- Run meanshift to find all locations

### Conditional Regression Forest

- Conditional regression tree works alike.

- For **training**:

  - Compute a probability for a concrete head pose

  - For each head pose divide the training set in disjoint subsets according to the pose

  - Train a regression forest for each subset 

- For **testing**:

  - Estimate the probabilities for each head pose

  - Select trees from different regression forests

  - Estimate the density function for all facial feature points.

  - Finalize the exact poition by clustering over all feature candidate votes for a given facial feature point. (e.g., by meanshift)

### Experiments and results

- Training set: 
  - 13233 face images from LFW Database
  - 10 annotated facial feature points per face image
- Training
  - Maximum tree depth = 20
  - 2500 splitting candidates and 25 thresholds per split
  - 1500 images to train each tree
  - 200 patches per image (20 * 20 pixels).
  - For head pose two different subsets with 3 and 5 head poses are generated (accuracy 72,5%)
  - Required time for face detection and head pose estimation is 33 ms.

- Results

  ![截屏2021-02-19 16.35.41](/Users/EckoTan/Library/Application Support/typora-user-images/截屏2021-02-19 16.35.41.png)



## CNN based models

**Stacked Hourglass Network** [^2]

- Fully-convolutional neural network

- Repeated down- and upsampling + shortcut connections
- Based on RGB face image, produce one heatmap for each landmark 
- Heatmaps are transformed into numerical coordinates using DSNT

![截屏2021-02-19 16.51.43](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-19%2016.51.43.png)







[^1]: M. Dantone, J. Gall, G. Fanelli and L. Van Gool, "Real-time facial feature detection using conditional regression forests," 2012 IEEE Conference on Computer Vision and Pattern Recognition, Providence, RI, USA, 2012, pp. 2578-2585, doi: 10.1109/CVPR.2012.6247976.
[^2]: Newell, A., Yang, K., & Deng, J. (2016). Stacked hourglass networks for human pose estimation. *Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)*, *9912 LNCS*, 483–499. https://doi.org/10.1007/978-3-319-46484-8_29

