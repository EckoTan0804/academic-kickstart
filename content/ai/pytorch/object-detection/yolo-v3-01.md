---
# Title, summary, and position in the list
linktitle: "YOLOv3 (1)"
summary: ""
weight: 501

# Basic metadata
title: "YOLOv3 (1): How YOLO works"
date: 2020-11-08
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "Object Detection", "YOLO"]
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
        parent: object-detection
        weight: 1
---

## How YOLO works?

### Fully Convolutional Neural Network

YOLO (You Only Look Once) makes use of only convolutional layers, making it a **fully convolutional network (FCN)**.

- 75 convolutional layers, with skip connections and upsampling layers
- NO form of pooling is used
- a convolutional layer with stride 2 is used to downsample the feature maps.
- invariant to the size of the input image
  - However, in practice, we might want to stick to a constant input size

### Interpreting the output

Typically, (as is the case for all object detectors) the features learned by the convolutional layers are passed onto a classifier/regressor which makes the detection prediction (coordinates of the bounding boxes, the class label.. etc).

In YOLO, the prediction is done by using a convolutional layer which uses $1 \times 1$ convolutions.

The first thing to notice is our **output is a feature map**. In YOLO v3 (and it's descendants), the way we interpret this prediction map is that each cell can predict a fixed number of bounding boxes.

**Depth-wise, we have $(B \times (5 + C))$ entries in the feature map**

- $B$: number of bounding boxes each cell can predict
  - YOLO v3 predicts 3 bounding boxes for every cell.
- Each of the bounding boxes have $5 + C$ attributes, which describe 
  - the center coordinates ($x, y$ coordinates)
  - the dimensions (height and width)
  - the objectness score
  - $C$ class confidences for each bounding box. 



Let's take a look at an example:

The input imge is $416 \times 416$, stride of the network is 32. Thus, the dimensions of the feature map will be $13 \times 13$. We then divide the input image into $13 \times 13$ cells.

![yolo-5](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/yolo-5.png)

Then, the cell (*on the input image*) containing the center of the ground truth box of an object is chosen to be the one responsible for predicting the object. In the image, it is the cell which marked <span style="color:red">red</span>, which contains the center of the ground truth box (marked yellow).

The red cell is the 7th cell in the 7th row on the grid. We now assign the 7th cell in the 7th row **on the feature map** (corresponding cell on the feature map) as the one responsible for detecting the dog.

Now, this cell can predict three bounding boxes. Which one will be assigned to the dog's ground truth label? In order to understand that, we must wrap out head around the concept of anchors.

### Anchor Boxes

It might make sense to predict the width and the height of the bounding box, but in practice, that leads to *unstable gradients* during training :cry:. Instead, most of the modern object detectors predict log-space transforms, or simply offsets to pre-defined default bounding boxes called **anchors**.

YOLO v3 has three anchors, which result in prediction of three bounding boxes per cell.

Coming back to our earlier question in the example above, the bounding box responsible for detecting the dog will be the one whose anchor has the **highest IoU (Intersection over Union)** with the ground truth box.

### Making Predictions

The network output is transformed with the following formulars to obtain bounding b ox predictions:
$$
\begin{aligned}
b\_{x} &=\sigma\left(t\_{x}\right)+c\_{x} \\\\
b\_{y} &=\sigma\left(t\_{y}\right)+c\_{y} \\\\
b\_{w} &=p\_{w} e^{t\_{w}} \\\\
b\_{h} &=p\_{h} e^{t\_{h}}
\end{aligned}
$$

- $t\_x, t\_y, t\_w, t\_h$: network outputs
- $c\_x, c\_y$: top-left coordinates of the grid cell
- $p\_w, p\_h$: anchors dimensions for the box
- $b\_x, b\_y, b\_w, b\_h$: $(x, y)$ center coordinates, width and height of our prediction
- $\sigma$: sigmoid function, forces the value of the output to be between 0 and 1.

#### Coordinates

Normally, YOLO doesn't predict the absolute coordinates of the bounding box's center. It predicts offsets which are:

- Relative to the top left corner of the grid cell which is predicting the object.
- Normalised by the dimensions of the cell from the feature map, which is, 1.

For example, consider the case of our dog image. If the prediction for center is $(0.4, 0.7)$, then this means that the center lies at $(6.4, 6.7)$ on the $13 \times 13$ feature map. (Since the top-left coordinates of the red cell are $(6, 6)$).

#### **Dimensions of the Bounding Box**

 The dimensions of the bounding box are predicted by 

- applying a log-space transform to the output 
- then multiplying with an anchor.

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/yolo-regression-1.png" title="Transform the detector output to give the final prediction (Src: *http://christopher5106.github.io/*)" numbered="true" >}}

#### Objectness Score

Object score represents the **probability that an object is contained inside a bounding box**. It should be nearly 1 for the red and the neighboring grids, whereas almost 0 for, say, the grid at the corners.

The objectness score is also passed through a sigmoid, as it is to be interpreted as a probability.

#### **Class Confidences**

Class confidences represent the **probabilities of the detected object belonging to a particular class** (Dog, cat, banana, car etc). 

Before v3, YOLO used to **softmax** the class scores. However, that design choice has been dropped in v3, and authors have opted for using **sigmoid** instead.

- The reason is that Softmaxing class scores assume that the classes are *mutually exclusive*. In simple words, if an object belongs to one class, then it's guaranteed it cannot belong to another class. This is true for COCO database on which we will base our detector.
- However, this assumptions may not hold when we have classes like *Women* and *Person*. This is the reason that authors have steered clear of using a Softmax activation.

### Prediction across different scales

- YOLO v3 makes prediction across 3 different scales. The detection layer is used make detection at feature maps of three different sizes, having **strides 32, 16, 8** respectively. This means, with an input of $416 \times 416$, we make detections on scales $13 \times 13$, $26 \times 26$ and $52 \times 52$.

- The network downsamples the input image until the first detection layer, where a detection is made using feature maps of a layer with stride 32. Further, layers are upsampled by a factor of 2 and concatenated with feature maps of a previous layers having identical feature map sizes. Another detection is now made at layer with stride 16. The same upsampling procedure is repeated, and a final detection is made at the layer of stride 8.

- At each scale, each cell predicts 3 bounding boxes using 3 anchors, making the total number of anchors used 9. (The anchors are different for different scales)

![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/yolo_Scales-1.png)

The authors report that this helps YOLO v3 get better at detecting small objects. Upsampling can help the network learn fine-grained features which are instrumental for detecting small objects.

### Output Processing

For an image of size 416 x 416, YOLO predicts $((52 \times 52) + (26 \times 26) + 13 \times 13)) \times 3 = 10647$ bounding boxes.

However, in case of our image, there's only one object, a dog. How do we reduce the detections from 10647 to 1 bounding box?

We can handle the situation like this:

1. **Thresholding by Object Confidence**

   Filter boxes based on their objectness score. Generally, boxes having scores below a threshold are ignored.

2. **Non-maximum Suppression**

   Clean up redundant detections and make sure that YOLO detects the object just once

   {{< figure src="https://www.jeremyjordan.me/content/images/2018/07/Screen-Shot-2018-07-10-at-9.46.29-PM.png" title="Non-max suppression example. (Src: [An overview of object detection: one-stage methods.](https://www.jeremyjordan.me/object-detection-one-stage/))" numbered="true" >}}

## Reference

[Series: YOLO object detector in PyTorch](https://blog.paperspace.com/tag/series-yolo/)





















