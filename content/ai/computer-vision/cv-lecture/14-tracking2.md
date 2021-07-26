---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 114

# Basic metadata
title: "Tracking 2"
date: 2021-07-20
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
        weight: 14
---

## Multi-Camera Systems

### Type of multi-camera systems

- **Stereo-camera system** (narrow baseline)

  ​	![截屏2021-07-20 17.13.58](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-20%2017.13.58-20210720171721568.png)

  - Close distance and equal orientation
  - An object’s appearance is almost the same in both cameras
  - Allows for calculation of a dense disparity map

- **Wide-baseline multi-camera system**

  ​	![截屏2021-07-20 17.15.59](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-20%2017.15.59.png)

  - Arbitrary distance and orientation, overlapping field of view

  - An object’s appearance is different in each of the cameras

    ![截屏2021-07-20 17.16.15](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-20%2017.16.15.png)

  - Allows for 3D localization of objects in the joint field of view

- **Multi-camera network**

  ​	<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-20%2017.16.55.png" alt="截屏2021-07-20 17.16.55" style="zoom:67%;" />

  - Non-overlapping field of view
  - An object’s appearance differs strongly from one camera to another

### 3D to 2D projection: Pinhole Camera Model

Summary:

![截屏2021-07-24 18.49.45](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-24%2018.49.45.png)

![截屏2021-07-20 17.19.45](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-20%2017.19.45.png)
$$
z^{\prime} = -f
$$

$$
\frac{y^{\prime}}{-f}=\frac{y}{z} \Rightarrow y^{\prime}=\frac{-f y}{z}
$$

$$
\frac{x^{\prime}}{-f}=\frac{x}{z}  \Rightarrow  x^{\prime}=\frac{-f x}{z}
$$

Pixel coordinates $(u, v)$ of the projected points on **image plane**

![截屏2021-07-20 18.24.21](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-20%2018.24.21.png)
$$
\begin{array}{l}
\boldsymbol{u}=\boldsymbol{k}\_{u} \boldsymbol{x}^{\prime}+\boldsymbol{u}\_{\mathrm{0}} \\\\
\boldsymbol{v}=-\boldsymbol{k}\_{v} \boldsymbol{y}^{\prime}+\boldsymbol{v}\_{\mathrm{0}}
\end{array}
$$
where $k\_u$ and $k\_v$ are **scaling factors** which denote the ratio between world and pixel coordinates.

In matrix formulation:
$$
\left(\begin{array}{l}
u \\\\
v
\end{array}\right)=\left(\begin{array}{cc}
k\_{u} & 0 \\\\
0 & -k\_{v}
\end{array}\right)\left(\begin{array}{l}
x^{\prime} \\\\
y^{\prime}
\end{array}\right)+\left(\begin{array}{l}
u\_{0} \\\\
v\_{0}
\end{array}\right)
$$
**Perspective Projection**

- internal camera parameters
  $$
  \begin{array}{l}
  \alpha\_{u}=k\_{u} f \\\\
  \alpha\_{v}=-k\_{v} f \\\\
  u\_{0} \\\\
  v\_{0}
  \end{array}
  $$

  - have to be known to perform the projection
  - they depend on the camera only
  - Perform calibration to estimate

#### Calibration

**Intrinsics parameters**: describe the optical properties of each camera (“the camera model”)

- $f$: focal length
- $c\_x, c\_y$: the principal point ("optical center"), sometimes also denoted as $u\_0, v\_0$
- $K\_1, \dots, K\_n$: distortion parameters (radial and tangential)

**Extrinsic parameters**: describe the location of each camera with respect to a global coordinate system

- $\mathbf{T}$: translation vector
- $\mathbf{R}$: $3 \times 3$ rotation matrix

Transformation of world coordinate of point $p^* = (x, y, z)$  to camera coordinate $p$:
$$
p = \mathbf{R} (x, y, z)^T + \mathbf{T}
$$
Calibration steps

1. For each camera: A calibration target with a known geometry is captured from multiple views
2. The corner points are extracted (semi-)automatically
3. The locations of the corner points are used to estimate the intrinsics iteratively
4. Once the intrinsics are known, a fixed calibration target is captured from all of the camerasextrinsics

### Triangulation

![截屏2021-07-20 19.14.22](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-20%2019.14.22.png)

- Assumption: the object location is known in multiple views
- Ideally: The intersection of the lines-of-view determines the 3D location
- Practically: least-squares approximation