---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 106

# Basic metadata
title: "Face Recognition: Features"
date: 2021-02-15
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
        weight: 6
---

## Local Appearance-based Face Recognition

🎯 Objective: To mitigate the effect of expression, illumination, and occlusion variations by performing local analysis and by fusing the outputs of extracted local features at the feature or at the decision level.

Some popular facial descriptions achieving good results 

- Local binary Pattern Histogram (LBPH)
- Gabor Feature
- Discrete Cosine Transform (DCT)
- SIFT 
- etc.

### Local binary Pattern Histogram (LBPH)[^1]

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2011.10.39.png" alt="截屏2021-02-16 11.10.39" style="zoom:80%;" />

- Divide image into cells

- Compare each pixel to each of its neighbors

  - Where the pixel's value is greater than the threshold value (e.g., center pixel in this example), write "1"
  - Otherwise, write "0"

  $\rightarrow$ gives a binary number

- Convert binary into decimal
- Compute the histogram, over the cell
- Use the histogram for classification
  - SVM 
  - Histogram-distances

> Tutorials and explanation:
>
> - [Face Recognition: Understanding LBPH Algorithm](https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b)
> - [how is the LBP |Local Binary Pattern| values calculated? ~ xRay Pixy](https://www.youtube.com/watch?v=h-z9-bMtd7w)



### **High dim. dense local Feature Extraction**

- Computing features densely (e.g. on overlapping patches in many scales in the image)
- Problem: very very high dimensionality!!!
- Solution: Encode into a compact form
  - Bag of Visual Word (BoVW) model
  - Fisher encoding

#### Fisher Vector Encoding

- Aggregates feature vectors into a compact representation 
- Fitting a parametric generative model (e.g. Gaussian Mixture Model)
- Encode derivative of the likelihood of model w.r.t its parameters

![截屏2021-02-16 11.38.19](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2011.38.19.png)



## Face recognition across pose (Alignment)

Problem

- Different view-point / head orientation

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2011.40.44.png" alt="截屏2021-02-16 11.40.44" style="zoom:80%;" />

- Recoginition results degrade, when images of different head orientation have to be matched 😭

**Major directions to address the face recognition across pose Probelm**

- Geometric pose normalization (image affine warps)
- 2D specific pose models, image rendering at pixel or feature level (2D+3D approaches)
- 3D face Model fitting

### Pose Normalization

#### 💡 **Idea**

- Find several facial features (mesh) 
- Use complete mesh to normalize face

Here we will use **2D Active Appearance Models**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2011.51.52.png" alt="截屏2021-02-16 11.51.52" style="zoom:80%;" />

- A texture and shape-based parametric model

- Efficient fitting algorithm: **Inverse compositional (IC)** algorithm

#### Model and fitting

Independent shape and appearance model
$$
\begin{array}{c}
\text{shape:} \quad s=\left(x\_{1}, y\_{1}, x\_{2}, y\_{2}, \cdots, x\_{v}, y\_{v}\right)^{T}=s\_{0}+\sum\_{i=1}^{n} p\_{i} s\_{i} \\\\
\text{appearance:} \quad A(x)=A\_{0}(x)+\sum\_{i=1}^{m} \lambda\_{i} A\_{i}(x) \quad \forall x \in s\_{0}
\end{array}
$$
Fitting goal:
$$
\arg \min \_{p, \lambda} \sum\_{x \in s\_{0}}\left[A\_{0}(x)+\sum\_{i=1}^{m} \lambda_{i} A\_{i}(x)-I(W(x ; p))\right]^{2}
$$
Fitting examples

- Fitted mesh

  ![截屏2021-02-16 12.02.54](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2012.02.54.png)

- Mismatched mesh

  ![截屏2021-02-16 12.03.27](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2012.03.27.png)

Fitted modal can be used to warp image to frontal pose (e.g. using piecewise affine transformation of mesh triangles)

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2012.13.08.png" title="Faces with different poses from FERET data base and their pose- aligned images" numbered="true" >}}

#### Results

- Much better results under pose variations compared to simple affine transform
- Different warping functions can be used
  - Piecewise affine transformation worked best
- Approach works well with local-DCT-based approach
  - but not so well with holistic approaches, such as Eigenfaces (PCA) 🤪

## Face Recogntion using 3D Models[^2]

- A method for face recognition across variations in pose and illumination.
- Simulates the process of image formation in 3D space.
- Estimates 3D shape and texture of faces from single images by fitting a statistical morphable model of 3D faces to images.
- Faces are represented by model parameters for 3D shape and texture.

#### Model-based Recognition

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2012.19.23.png" alt="截屏2021-02-16 12.19.23" style="zoom:67%;" />

#### Face vectors

The morphable face model is based on a vector space representation of faces that is constructed such that **any combination of shape and texture vectors $S\_i$ and $T\_i$ describes a realistic human face**:
$$
S=\sum_{i=1}^{m} a_{i} S_{i} \quad T=\sum_{i=1}^{m} b_{i} T_{i}
$$
The definition of shape and texture vectors is based on a reference face $\mathbf{I}\_0$.

The location of the vertices of the mesh in Cartesian coordinates is $(x\_k, y\_k, z\_k)$ with colors $(R\_k, G\_k, B\_k)$

Reference shape and texture vectors are defined by:
$$
\begin{array}{l}
S\_{0}=\left(x\_{1}, y\_{1}, z\_{1}, x\_{2}, \ldots, x\_{n}, y\_{n}, z\_{n}\right)^{T} \\\\
T\_{0}=\left(R\_{1}, G\_{1}, B\_{1}, R\_{2}, \ldots, R\_{n}, G\_{n}, B\_{n}\right)^{T}
\end{array}
$$
To encode a novel scan $\mathbf{I}$, the flow field from $\mathbf{I}\_0$ to $\mathbf{I}$ is computed.

#### PCA

- PCA is performed on the set of shape and texture vectors separately.

- Eigenvectors form an orthogonal basis:
  $$
  \mathbf{S}=\overline{\mathbf{s}}+\sum\_{i=1}^{m-1} \alpha\_{i} \cdot \mathbf{s}\_{i}, \quad \mathbf{T}=\overline{\mathbf{t}}+\sum\_{i=1}^{m-1} \beta\_{i} \cdot \mathbf{t}\_{i}
  $$

- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2020.36.08.png" alt="截屏2021-02-16 20.36.08" style="zoom:67%;" />

### Model-based Image Analysis

- 🎯 Goal: find shape and texture coefficients describing a 3D face model such that rendering produces an image $\mathbf{I}\_{\text{model}}$ that is as similar as possible to $\mathbf{I}\_{\text{input}}$

- For initialization 7 facial feature points, such as the corners of the eyes or tip of the nose, should be labelled manually

  ![截屏2021-02-16 20.38.43](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2020.38.43.png)

- Model fitting: Minimize
  $$
  E\_{I}=\sum\_{x, y}\left\\|\mathbf{I}\_{\text {input }}(x, y)-\mathbf{I}\_{\text {model }}(x, y)\right\\|^{2}
  $$

  - Shape, texture, transformation, and illumination are optimized for the entire face and refined for each segment.
  - Complex iterative optimization procedure



## Databases

- Necessary to develop and improve algorithms
- Provide common testbeds and benchmarks which allow for comparing different approaches
- Different databases focus on different problems

Well-known databases for face recognition

- FERET
- FRVT
- FRGC
- CMU-PIE
- BANCA
- XM2VTS
- ...

### Observations

- One 3-D image is *more powerful* for face recognition than one 2- D image.
- One high resolution 2-D image is *more powerful* for face recognition than one 3-D image.
- Using 4 or 5 well-chosen 2-D face images is *more powerful* for face recognition than one 3-D face image or multi-modal 3D+2D face.

#### Wild Face Datasets

#### **Labeled Faces In the Wild Dataset (LFW)**

- Face Verification: Given a pair of images specify whether they belong to the same person

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2020.44.55.png" alt="截屏2021-02-16 20.44.55" style="zoom:80%;" />

- 13K images, 5.7K people
- Standard benchmark in the community
- Several test protocols depending upon availability of training data within and outside the dataset.

#### **YouTube Faces Dataset (YTF)**

- Video Face Verification: Given a pair of videos specify whether they belong to the same person

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-02-16%2020.46.03.png" alt="截屏2021-02-16 20.46.03" style="zoom:80%;" />

- 3425 videos, 1595 people

- Standard benchmark in the community

- Wide pose, expression and illumination variation











[^1]: T. Ahonen, A. Hadid and M. Pietikainen, "Face Description with Local Binary Patterns: Application to Face Recognition," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 28, no. 12, pp. 2037-2041, Dec. 2006, doi: 10.1109/TPAMI.2006.244.
[^2]: V. Blanz and T. Vetter, "Face recognition based on fitting a 3D morphable model," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 25, no. 9, pp. 1063-1074, Sept. 2003, doi: 10.1109/TPAMI.2003.1227983.
