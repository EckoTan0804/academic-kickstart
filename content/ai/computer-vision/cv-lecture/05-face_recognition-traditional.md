---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 15

# Basic metadata
title: "Face Recognition: Traditional Approaches"
date: 2021-02-04
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Lecture", "Face", "Face Recognition"]
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
        weight: 5
---

## Face Recognition for Human-Computer Interaction (HCI)

### Main Problem

> The variations between the images of the same face due to illumination and viewing direction are almost always larger than image variations due to change in face identity.
>
> -- Moses, Adini, Ullman, ECCV‘94

![截屏2021-02-04 23.57.03](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-04%2023.57.03.png)

### Closed Set vs. Open Set Identification

- **Closed-Set Identification**
  - The system reports which person from the gallery is shown on the test image: Who is he?
  - Performance metric: Correct identification rate
- **Open-Set Identification**
  - The system first decides whether the person on the test image is a known or unknown person. If he is a known person who he is?
  - Performance metric
    - **False accept**: The invalid identity is accepted as one of the individuals in the database.
    - **False reject**: An individual is rejected even though he/she is present in the database.
    - **False classify**: An individual in the database is correctly accepted but misclassified as one of the other individuals in the training data

###Authentication/Verification

A person claims to be a particular member. The system decides if the test image and the training image is the same person: Is he who he claims he is?

Performance metric: 

- False Reject Rate (FRR): Rate of rejecting a valid identity
- False Accept Rate (FAR): Rate of incorrectly accepting an invalid identity.

## Feature-based (Geometrical) approaches

"Face Recognition: Features versus Templates" [^1]

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-05 00.12.27.png" alt="截屏2021-02-05 00.12.27" style="zoom:67%;" />

- Eyebrow thickness and vertical position at the eye center position
- A coarse description of the left eyebrow‘s arches
- Nose vertical position and width
- Mouth vertical position, width, height upper and lower lips
- Eleven radii describing the chin shape
- Face width at nose position
- Face width halfway between nose tip and eyes

### Classification

**Nearest neighbor classifier** with **Mahalanobis distance** as the distance metric:
$$
\Delta_{j}(x)=\left(x-m_{j}\right)^{T} \Sigma^{-1}\left(x-m_{j}\right)
$$

- $x$: input face image
- $m\_j$: average vector representing the $j$-th person
- $\Sigma$: Covariance matrix

Different people are characterized only by their average feature vector.

The distribution is common and estimated by using all the examples in the training set.

## Appearance-based approaches

Can be either

- **[holistic](#holistic-appearance-based-approaches)** (process the whole face as the input), or
- [**local / fiducial**](#local-appearance-based-approach) (process facial features, such as eyes, mouth, etc. seperately)

![截屏2021-02-08 10.28.36](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-08%2010.28.36.png)

Processing steps: align faces with facial landmarks

- Use manually labeled or automatically detected eye centers
- Normalize face images to a common coordination, remove translation,, rotation and scaling factors
- Crop off unnecessary background

![截屏2021-02-08 10.30.38](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-08%2010.30.38.png)

## Holistic appearance-based approaches

### Eigenfaces

#### 💡 Idea

A face image defines a point in the high dimensional image space.

Different face images share a number of similarities with each other

- They can be described by a relatively low dimensional subspace
- Project the face images into an appropriately chosen subspace and perform classification by similarity computation (distance, angle)
  - Dimensionality reduction procedure used here is called {{<hl>}}**Karhunen-Loéve transformation**{{</hl>}} or {{<hl>}}**principal component analysis (PCA)**{{</hl>}}

#### Objective

Find the vectors that best account for the distribution of face images within the entire image space

#### PCA

> For more details see: [Principle Component Analysis (PCA)]({{< relref "../../machine-learning/unsupervised/PCA.md" >}})

- Find direction vectors so as to minimize the average projection error
- Project on the linear subspace spanned by these vectors
- Use covariance matrix to find these direction vectors
- Project on the largest K direction vectors to reduce dimensionality

PCA for eigenfaces:

![截屏2021-02-09 11.11.40](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-09%2011.11.40.png)
$$
\begin{array}{l}
    Y=\left[y\_{1}, y\_{2}, y\_{3}, \ldots, y\_{K}\right] \\\\
    m=\frac{1}{K}\sum y \\\\
    C=(Y-m)(Y-m)^{T} \\\\
    D=U^{T} C U \\\\
    \Omega=U^{\top}(y-m)
\end{array}
$$
where

- $y$: Face image
- $Y$: Face matrix
- $m$: Mean face
- $C$: Covariance matrix
- $D$: Eigenvalues
- $U$: Eigenvectors

- $\Omega$: Representation coefficients

#### Training

- Acquire initial set of face images (training set):
  $$
  Y = [y\_1, y\_2, \dots, y\_K]
  $$

- Calculate the eigenfaces/eigenvectors from the training set, keeping only the $M$ images/vectors corresponding to the highest eigenvalues
  $$
  U = (u\_1, u\_2, \dots, u\_M)
  $$

- Calculate representation of each known individual $k$ in face space
  $$
  \Omega\_k = U^T(y\_k - m)
  $$

#### Testing

- Project input new image *y* into face space
  $$
  \Omega = U^T(y - m)
  $$

- Find most likely candidate class $k$ by distance computation
  $$
  \epsilon\_k = \\|\Omega - \Omega\_k\\| \quad \text{for all } \Omega\_k
  $$

#### **Projections onto the face space**

- Principal components are called “**eigenfaces**” and they span the “face space”.

- Images can be reconstructed by their projections in face space:

$$
Y\_f = \sum\_{i=1}^{M} \omega\_i u\_i
$$

​		Appearance of faces in face-space does not change a lot

- Difference of mean-adjusted image $(Y-m)$ and projection $Y\_f$ gives a measure of *„faceness“*

  - Distance from face space can be used to detect faces

- Different cases of projections onto face space

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-09%2011.38.26.png" alt="截屏2021-02-09 11.38.26" style="zoom:80%;" />

  - Case 1: Projection of a *known* individual

    $\rightarrow$ Near face space ($\epsilon < \theta\_{\delta}$) and near known face $\Omega\_k$ ($\epsilon\_k < \theta\_{\epsilon}$)

  - Case 2: Projection of an *unkown* individual

    $\rightarrow$ Near face space, far from reference vectors

  - Case 3 and 4: not a face (far from face space)

#### PCA for face matching and recognition

![截屏2021-02-09 11.53.26](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-09%2011.53.26.png)

- Projects all faces onto a **universal** eigenspace to “encode” via principal components
- Uses inverse-distance as a similarity measure $S(p,g)$ for matching & recognition

#### Problems and shortcomings

- Eigenfaces do NOT distinguish between shape and appearance

- PCA does NOT use class information

  - PCA projections are optimal for reconstruction from a low dimensional basis, they may not be optimal from a discrimination standpoint

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-09%2011.58.01.png" alt="截屏2021-02-09 11.58.01" style="zoom:67%;" />



### Fisherface

#### Linear Discriminant Analysis (LDA)

> For more details about LDA, see: [LDA Summary]({{< relref "../../machine-learning/non-parametric/LDA-summary.md" >}}))

- A.k.a. **Fischer‘s Linear Discriminant**
- Preserves separability of classes
- Maximizes ratio of projected between-classes to projected within-class scatter

$$
W\_{\mathrm{fld}}=\arg  \underset{W}{\max } \frac{\left|W^{T} S\_{B} W\right|}{\left|W^{T} S\_{W} W\right|}
$$

Where

- $S\_{B}=\sum\_{i=1}^{c}\left|x\_{i}\right|\left(\mu\_{i}-\mu\right)\left(\mu\_{i}-\mu\right)^{T}$: Between-class scatter
  - $c$: Number of classes
  - $\mu\_i$: mean of class $X\_i$
  - $|X\_i|$: number of samples of $X\_i$
- $S\_{W}=\sum\_{i=1}^{c} \sum\_{x\_{k} \in X\_{i}}\left(x\_{k}-\mu\_{i}\right)\left(x\_{k}-\mu\_{i}\right)^{T}$: Within-class scatter

**LDA vs. PCA**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-09%2012.25.35.png" alt="截屏2021-02-09 12.25.35" style="zoom:67%;" />

#### LDA for Fisherfaces

Fisher’s Linear Discriminant

- projects away the within-class variation (lighting, expressions) found in training set
- preserves the separability of the classes.



## Local appearance-based approach

Local vs Holistic approaches:

- Local variations on the facial appearance (different expression,occlsion, lighting) 
  - lead to modifications on the entire representation in the holistic approaches
  - while in local approaches ONLY the corresponding local region is effected
- Face images contain different statistical illumination (high frequency at the edges and low frequency at smooth regions). It's easier to represent the varying statistics linearly by using local representation.
- Local approaches facilitate the weighting of each local region in terms of their effect on face recognition.

### Modular Eigen Spaces

Classification using fiducial regions instead of using entire face [^2].

![截屏2021-02-09 12.59.14](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-09%2012.59.14.png)

### Local PCA (Modular PCA)

- Face images are divided into $N$ smaller sub-images

- PCA is applied on each of these sub-images

  ![截屏2021-02-09 13.01.08](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-09 13.01.08.png)

- Performed **better** than global PCA on large variations of illumination and expression

- No imporvements under variation of pose

### Local Feature based

🎯 Objective: To mitigate the effect of expression, illumination, and occlusion variations by performing local analysis and by fusing the outputs of extracted local features at the feature or at the decision level.

#### Gabor Filters

#### Elastic Bunch Graphs (EBG)

#### Local Binary Pattern (LBP) Histogram





















[^1]: http://cbcl.mit.edu/people/poggio/journals/brunelli-poggio-IEEE-PAMI-1993.pdf
[^2]: Pentland, Moghaddam and Starner, "View-based and modular eigenspaces for face recognition," *1994 Proceedings of IEEE Conference on Computer Vision and Pattern Recognition*, Seattle, WA, USA, 1994, pp. 84-91, doi: 10.1109/CVPR.1994.323814.

