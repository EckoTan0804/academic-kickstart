---
# Title, summary, and position in the list
linktitle: Evaluation Metrics
summary: ""
weight: 1001

# Basic metadata
title: "Evaluation Metrics for Object Detection"
date: 2020-11-12
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Object Detection"]
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
        parent: object-detection
        weight: 1
---

## Precision & Recall

Confusion matrix:

![Image result for true positive false positive](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*CPnO_bcdbE8FXTejQiV2dg.png)

- **Precision**: measures how accurate is your predictions. i.e. the percentage of your predictions are correct.
  $$
  \text{precision} = \frac{TP}{TP + FP}
  $$
  

- **Recall**: measures how good you find all the positives. 
  $$
  \text{recall} = \frac{TP}{TP + FN}
  $$

> More see: [Evaluation]({{< relref "../../machine-learning/ml-fundamentals/evaluation.md" >}})

## IoU (Intersection over union)

- IoU measures the overlap between 2 boundaries. 

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/0*VnvOCo9NkWG705F3.png" alt="Image for post" style="zoom:67%;" />

- We use that to measure how much our predicted boundary overlaps with the ground truth (the real object boundary).

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*FrmKLxCtkokDC3Yr1wc70w.png" alt="Image for post" style="zoom:80%;" />



## AP (Average Precision)

Let’s create an over-simplified example in demonstrating the calculation of the average precision. In this example, the whole dataset contains 5 apples only. 

We collect all the predictions made for apples in all the images and rank it in descending order according to the predicted confidence level. The second column indicates whether the prediction is correct or not. In this example, the prediction is correct if IoU ≥ 0.5.

![Image for post](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*9ordwhXD68cKCGzuJaH2Rg.png)

Let's look at the 3rd row:

- **Precision**: proportion of TP (= 2/3 = 0.67)
- **Recall**: proportion of TP out of the possible positives (= 2/5 = 0.4)

Recall values increase as we go down the prediction ranking. However, precision has a *zigzag* pattern — it goes down with false positives and goes up again with true positives.

![Image for post](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*ODZ6eZMrie3XVTOMDnXTNQ.jpeg)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*VenTq4IgxjmIpOXWdFb-jg.png" alt="Image for post" />

The general definition for the **Average Precision (AP)** is finding the **area under the precision-recall curve** above.
$$
\mathrm{AP}=\int\_{0}^{1} p(r) d r
$$

### Smoothing the Precision-Recall-Curve

Before calculating AP for the object detection, we often **smooth** out the zigzag pattern first: at each recall level, we replace each precision value with the maximum precision value to the right of that recall level.

![Image for post](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*pmSxeb4EfdGnzT6Xa68GEQ.jpeg)

The orange line is transformed into the green lines and the curve will decrease monotonically instead of the zigzag pattern.

| Before smoothing                                             | After smoothing                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*VenTq4IgxjmIpOXWdFb-jg.png" alt="Image for post" style="zoom: 67%;" /> | <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*zqTL1KW1gwzion9jY8SjHA-20201112121009694.png" alt="Image for post" style="zoom:67%;" /> |

Mathematically, we replace the precision value for recall $\tilde{r}$ with the maximum precision for any recall $\geq \tilde{r}$.
$$
p\_{\text {interp}}(r)=\max\_{\tilde{r} \geq r} p(\tilde{r})
$$

### **Interpolated AP**

PASCAL VOC is a popular dataset for object detection. In Pascal VOC2008, an average for the 11-point interpolated AP is calculated.

![Image for post](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*naz02wO-XMywlwAdFzF-GA.jpeg)

1. Divide the recall value from 0 to 1.0 into 11 points — 0, 0.1, 0.2, …, 0.9 and 1.0.

2. Compute the average of maximum precision value for these 11 recall values.
   $$
   \begin{aligned}
   A P &=\frac{1}{11} \sum\_{r \in\\{0.0, \ldots, 1.0\\}} A P\_{r} \\\\
   &=\frac{1}{11} \sum\_{r \in\\{0.0, \ldots, 1.0\\}} p\_{\text {interp}}(r)
   \end{aligned}
   $$

   - In our example:

     $AP = \frac{1}{11} \times (5 \times 1.0 + 4 \times 0.57 + 2 \times 0.5)$

However, this interpolated method is an approximation which suffers two issues. 

- It is less precise. 
- It lost the capability in measuring the difference for methods with low AP. 

Therefore, a different AP calculation is adopted after 2008 for PASCAL VOC.

### AP (Area under curve AUC)

For later Pascal VOC competitions, VOC2010–2012 samples the curve at all unique recall values (*r₁, r₂, …*), whenever the maximum precision value drops. With this change, we are **measuring the exact area under the precision-recall curve** after the zigzags are removed.

![Image for post](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*TAuQ3UOA8xh_5wI5hwLHcg.jpeg)

No approximation or interpolation is needed :clap:. Instead of sampling 11 points, **we sample $p(r\_i)$ whenever it drops and computes AP as the sum of the rectangular blocks**.
$$
\begin{array}{l}
p\_{\text {interp}}\left(r\_{n+1}\right)=\displaystyle{\max\_{\tilde{r} \geq r\_{n+1}}} p(\tilde{r}) \\\\
\mathrm{AP}=\sum\left(r\_{n+1}-r\_{n}\right) p\_{\text {interp}}\left(r\_{n+1}\right) 
\end{array}
$$
This definition is called the **Area Under Curve (AUC)**.

## Reference

- [**mAP (mean Average Precision) for Object Detection**](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173) :thumbsup:
- [The Confusing Metrics of AP and mAP for Object Detection / Instance Segmentation](https://medium.com/@yanfengliux/the-confusing-metrics-of-ap-and-map-for-object-detection-3113ba0386ef)
- [详解object detection中的mAP](https://zhuanlan.zhihu.com/p/56961620)​ :thumbsup:

