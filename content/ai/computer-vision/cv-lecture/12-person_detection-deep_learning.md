---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 112

# Basic metadata
title: "People Detection: Deep Learning Approaches"
date: 2021-07-16
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
        weight: 12
---

## Deep Learning for Object Detection

- People detections is a special case of object detection (one of the most challenging object classes to detect)
- Recently, most detectors are trained for the more challenging task of multi-object detection
  - Goal: Given an image, detect all instances of, say, 1000 different object classes
  - “Person” always one of the classes
- **Speed** is an issue
  - **Sliding Window**: Look at each position, each scale
  - **Cascades** look at each position too
    - They just take a shorter look at most positions/scales
  - **Region Proposals**: Avoid useless positions/scales from the beginning

### Region Proposals

- **💡Idea**
  - **Identify image regions that are likely to contain an object**
  - **Don’t care about the object class in the regions at this point**

- Characterization of a general object
  - Find “blobby” regions
  - Find connected regions that are somehow distinct from their surroundings

- Requirements

  - FAST!!!
  - High recall
  - Can allow a relatively high amount of false positives

- 2 main categories

  - **Grouping methods**

    - Generate proposals based on hierarchically grouping meaningful image regions

    - Often better localization

    - E.g. Selective search

      ![截屏2021-07-16 22.54.32](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-16%2022.54.32-20210719164944153.png)

  - **Window scoring methods**

    - Generate a large amount of windows
    - Use a quickly computed cue to discard unlikely windows (“objectness” measure)
    - Often faster

{{% alert note %}} 

For more details and comparison, see: [Overview of Region-based Object Detectors]({{< relref "../object-detection/overview-region-based-detectors.md" >}})

{{% /alert%}}

## R-CNN [^1]

### Idea and structure

![截屏2021-07-19 16.46.22](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2016.46.22.png)

![截屏2021-07-19 16.50.11](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2016.50.11.png)

### Training

1. Train AlexNet on ImageNet (1000 classes)

   ![截屏2021-07-19 16.53.30](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2016.53.30.png)

2. Re-initialize last layers to a different dimension (depending on the \#classes of the new classifier) and train new model

   ![截屏2021-07-19 16.54.29](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2016.54.29.png)

3. Train a classifier 

   - **Binary** SVMs (e.g. is human? yes/no) for each object class $\rightarrow$ $C$ SVMs in our case

   - The outputs of pool5 of the retrained AlexNet are used as features

     ![截屏2021-07-19 16.58.56](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2016.58.56.png)

4. Improve the region proposals

   - Use a regression model to improve the estimated locatin of the object

     - Input: features of proposed region (pool5)
     - output: x, y, width, height of the estimated region

     ![截屏2021-07-19 16.59.53](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2016.59.53.png)

### Downsides

1. **Speed**: Need to forward-pass **EACH** region proposal through entire CNN!!!

2. SVM & BBox regressor are trained after CNN is fixed
   - No simultaneous update/adaptation of CNN features possible

3. Complexity: multi-stage approach

Improvement:

- For 1: Can we make (part of) the CNN run only once for all proposals?

- For 2&3: Can we make the CNN perform these steps?

## Fast R-CNN [^2]

### Overview

![截屏2021-07-19 21.02.20](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.02.20.png)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.03.23.png" alt="截屏2021-07-19 21.03.23" style="zoom:80%;" />

### ROI pooling

- Conv layers don’t care about input size, FC layers do
- **ROI pooling**: warp the variable size ROIs into in a predefined fix size shape.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.14.14.png" alt="截屏2021-07-19 21.14.14" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*5V5mycIRNu-mK-rPywL57w-20210719211433781.gif" alt="Image for post" style="zoom:67%;" />

### End-to-end training

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.16.14.png" alt="截屏2021-07-19 21.16.14" style="zoom:70%;" />

- Instead of SVM & Regressor just add corresponding losses and train the system for both (multitask)
- Gradients can backprop. into feature layers through ROI pooling layers (just as with normal maxpool layers)
- End-to-end brings slight improvement 👏
- Softmax (integrated) loss slightly but consistently outperforms external classifier 👏

### **Fast R-CNN vs R-CNN**

Speed:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.21.53.png" alt="截屏2021-07-19 21.21.53" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.23.51.png" alt="截屏2021-07-19 21.23.51" style="zoom:67%;" />

### Downsides

- Majority of time is lost for region proposals

- Model is also not fully end-to-end: proposals come from “outside”

  (Can we include them in the CNN as well? 🤔)



## Faster R-CNN[^3]

### Overview

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*JQfhkHK6V8NRuh-97Pg4lQ-20210719212516440.png" alt="Image for post" style="zoom:67%;" />

### **Region Proposal Network (RPN)**

- Input: Feature map from larger conv network of size $C \times W \times H$

- Output
  - List of $p$ proposals
  - "Objectness" score of size $p \times 6$
    - $p \times 4$ coordinates (top-left and bottom-right $(x,y) $ coordinates) for bounding box
    - $p \times 2$ for objectness (with vs. without object) per location

- General approach:

  - Take a mini net (RPN) and slide it over the feature map (stepsize 1) 
  - At each position evaluate $k$ different window sizes for objectness 
  - Results in approx. $W \times H \times k$ windows/proposals

  ![Image for post](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*PszFnq3rqa_CAhBrI94Eeg-20210719214347181.png)

- Fully convolutional network
- **Anchors**: tackle the scale problem of the feature map
  - Initial reference boxes consisting of aspect ratio and scale, centered at sliding window
  - 3 scales and 3 aspect ratios = 9 anchors

- Layers
  - reg layer: regression of the reference anchor
  - cls layer: object/no object score

#### Loss

- Need a label for each anchor to train the objectness classification

- Labelling anchors

  - Positive: highest IoU with groundtruth *or* IoU > 0.7 (can be more than one)
    - Also store the association between anchor and groundtruth box
  - Negative: others, if their IoU < 0.3
  - Other anchors do not contribute to training

  $\rightarrow$ Convert to classification problem

- RPN multitask loss:
  $$
  L\left(\left\{p\_{i}\right\},\left\{t\_{i}\right\}\right)=\frac{1}{N\_{c l s}} \sum\_{i} L\_{c l s}\left(p\_{i}, p\_{i}^{*}\right) + \lambda \frac{1}{N\_{\text {reg }}} \sum\_{i} p\_{i}^{*} L\_{r e g}\left(t\_{i}, t\_{i}^{*}\right)
  $$

  - $N\_{cls}$: Batch size (256)
  - $N\_{reg}$: number of window positions ($\approx$ 2400)
  - $\lambda = 10$

### Training

#### As in paper

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.54.06.png" alt="截屏2021-07-19 21.54.06" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.54.33.png" alt="截屏2021-07-19 21.54.33" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.54.59.png" alt="截屏2021-07-19 21.54.59" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.55.21.png" alt="截屏2021-07-19 21.55.21" style="zoom:67%;" />

### Jointly

- Train everything in one go

- Combination of four losses 

  - objectness classification 
  - anchor regression
  - object class classification 
  - detection regression

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/fast_rcnn_loss.png" alt="fast_rcnn_loss" style="zoom: 67%;" />

> Why two regression losses?
>
> Anchor regression directly impacts the feature used for detection. Detection regression merely improves final localization

### Comparison between all the R-CNNs

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2021.58.56.png" alt="截屏2021-07-19 21.58.56" style="zoom:80%;" />



## SSD Detector [^4]

### Motivation

Thus far, deep multiclass detectors rely on variants of three steps:

- generate bounding boxes (proposals)
- resample pixels/features in boxes to uniform size
- apply high quality classifier

Can we avoid / speed up any of those steps to increase overall speed?

### Overview

![截屏2021-07-19 23.05.14](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2023.05.14.png)

- 💡**Core Idea: Use a set of fixed default boxes at each position in a feature map (similar to anchors)**
- Classify object class and box regression for each default box
- pply boxes at different layers in the ConvNet 
  - Use layers of different sizes
  - Avoids the need for rescaling

### Structure

![截屏2021-07-19 23.05.56](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-07-19%2023.05.56.png)

- Detectors at various stages with varying numbers of default boxes
- Resulting number of detections is fixed
- Reduced by non maximum suppression





[^1]: Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 580–587. https://doi.org/10.1109/CVPR.2014.81 ↩
[^2]: Girshick, R. (2015). Fast R-CNN. *Proceedings of the IEEE International Conference on Computer Vision*, *2015 International Conference on Computer Vision*, *ICCV 2015*, 1440–1448. https://doi.org/10.1109/ICCV.2015.169 ↩
[^3]: Ren, S., He, K., Girshick, R., & Sun, J. (2017). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *39*(6), 1137–1149. https://doi.org/10.1109/TPAMI.2016.2577031 ↩
[^4]: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu: “SSD: Single Shot MultiBox Detector”, 2016; [arXiv:1512.02325](http://arxiv.org/abs/1512.02325).

