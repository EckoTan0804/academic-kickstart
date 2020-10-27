---
# Title, summary, and position in the list
linktitle: "CNN Basics"
summary: ""
weight: 420

# Basic metadata
title: "Convolutional Neural Network (CNN) Basics"
date: 2020-08-19
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "CNN"]
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
        parent: cnn
        weight: 2

---

## Architecture Overview

All CNN models follow a similar architecture

![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/dobVrh3SGyqQraM2ogi-P3VK2K-LFsBm7RLO.png)

- [Input](#input-layer)
- [Convolutional layer (Cons-layer)](#convolutional-layer) + ReLU
- [Pooling layer (Pool-layer)](#pooling-layer)
- [Fully Connected layer (FC-layer)](#fully-connected-layer)
- Output



## Input

The input layer represents the **input image** into the CNN. Essentially, every image can be represented as a **matrix of pixel values**.

![8-gif.gif](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/8-gif.gif)

**[Channel](https://en.wikipedia.org/wiki/Channel_(digital_image))** is a conventional term used to refer to a certain component of an image.

- [Grayscale](https://en.wikipedia.org/wiki/Grayscale) image: has just one channel
- RGB images
  - Three channels: Red, Green, Blue
  - Imagine those as three 2d-matrices stacked over each other (one for each color), each having pixel values in the range 0 to 255.

> We can consider channel as **depth** of the image.
>
> <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="421px" viewBox="-0.5 -0.5 421 201" content="&lt;mxfile host=&quot;app.diagrams.net&quot; modified=&quot;2020-09-02T09:43:59.266Z&quot; agent=&quot;5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36&quot; etag=&quot;m-8WQ35sl3dUEUPa7ioz&quot; version=&quot;13.6.6&quot; type=&quot;device&quot;&gt;&lt;diagram id=&quot;GxF2ZYXmkwzIqcZl7oPp&quot; name=&quot;Page-1&quot;&gt;5ZdPb9sgGMY/jY+TbHAc99g4XXuptinadpwQvLFRsbEwiZ19+kGN7fhP1G1qlENPgQd4X/g9GIiHk7x5VKTMniUD4SGfNR7eeggFIYrMj1VOrbJeB62QKs5cp0HY8d/gRN+pB86gGnXUUgrNy7FIZVEA1SONKCXrcbe9FOOsJUlhJuwoEXP1J2c6a1WE8d3Q8AQ8zVxqjH0385x0vZ1QZYTJ+kzCDx5OlJS6LeVNAsLS68C04z5faO1npqDQfzPgMSm/SFLeoW8iONIfu+8vvw6fwjbKkYiDW7GbrD51COqMa9iVhNp6bWz28CbTuTC1wBRJVbbg97wBk2qz50IkUkj1OhwzAvGeGr3SSr5A11LIwsTbuPSgNDQX1xX0tMw+A5mDVifTxQ1YdYDdFkOdA/VgWND1yc686kXiNknaxx4wmoIj+Q9U8fWpriBm4TWp+hOq6OZUg6tT3ccU6DX3ahhPqIY3p7qaUXXn2ZStWbUe41ykdI7USUTwtDBVaiiB0TeWITeH7L1ryDljNs1myT8lDwWzbm19G14W2l0TyH8nU6ZbPZ6bEi94El7LkmhmSXv3fBhHpocPjm7syPrtowcKdm8fHJaqIFXF6diaKabWqu5RgXpwwGYvkjexnWFZLWDpNAWCaH4ch19i5TJ8ldwk7l2J0AVXuhCVPCgKbtT5S2QaaD0JFEwCaaJS0LNAr9b1y/5/N+OZm1so7ffl21TITzJiXpPi43xvUTQxZOmyj97ngzPV4c3bOjr8dcAPfwA=&lt;/diagram&gt;&lt;/mxfile&gt;" onclick="(function(svg){var src=window.event.target||window.event.srcElement;while (src!=null&amp;&amp;src.nodeName.toLowerCase()!='a'){src=src.parentNode;}if(src==null){if(svg.wnd!=null&amp;&amp;!svg.wnd.closed){svg.wnd.focus();}else{var r=function(evt){if(evt.data=='ready'&amp;&amp;evt.source==svg.wnd){svg.wnd.postMessage(decodeURIComponent(svg.getAttribute('content')),'*');window.removeEventListener('message',r);}};window.addEventListener('message',r);svg.wnd=window.open('https://viewer.diagrams.net/?client=1&amp;edit=_blank');}}})(this);" style="cursor:pointer;max-width:100%;max-height:201px;"><defs/><g><rect x="120" y="0" width="120" height="120" fill="#dae8fc" stroke="none" pointer-events="all"/><rect x="100" y="20" width="120" height="120" fill="#d5e8d4" stroke="none" pointer-events="all"/><rect x="80" y="40" width="120" height="120" fill="#f8cecc" stroke="none" pointer-events="all"/><rect x="0" y="80" width="80" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 100px; margin-left: 1px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Height</div></div></div></foreignObject><text x="40" y="106" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">Height</text></switch></g><rect x="100" y="160" width="80" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 180px; margin-left: 101px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Width</div></div></div></foreignObject><text x="140" y="186" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">Width</text></switch></g><path d="M 220 160 L 264.18 115.82" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 268.42 111.58 L 265.59 120.07 L 264.18 115.82 L 259.93 114.41 Z" fill="#000000" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><rect x="260" y="120" width="160" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 158px; height: 1px; padding-top: 140px; margin-left: 261px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Depth / Channel</div></div></div></foreignObject><text x="340" y="146" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">Depth / Channel</text></switch></g></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://desk.draw.io/support/solutions/articles/16000042487" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Viewer does not support full SVG 1.1</text></a></switch></svg>

## Convolutional Layer

### Convolution operation

- Extract features from the input image and produce feature maps
  1. Slide the convonlutional filter/kernel over the input image

  2. At every location, do **element-wise** matrix multiplication and sum the result. 

- This can preserve the spatial relationship between pixels by learning image features using small squares of input data :thumbsup:

#### 2D Convolution

Convolution operation in 2D using a $3\times3$ filter

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*cTEp-IvCCUYPTT0QpE3Gjg@2x.png" alt="Image for post" style="zoom: 38%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif" alt="Image for post" style="zoom:50%;" />

Another example:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/v2-15fea61b768f7561648dbea164fcb75f_b.gif" alt="十五）初识滤波之均值滤波- 知乎" style="zoom:80%;" />

#### 3D Convolution

In reality an image is represented as a 3D matrix with dimensions of height, width and depth, where depth corresponds to color channels (RGB). A convolution filter has a specific height and width, like $3 \times 3$ or $5 \times 5$, and by design it **covers the entire depth of its input** ($\text{depth}\_{\text{filter}} = \text{depth}\_{\text{input}}$).

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/0.gif" title="The convolution filter/kernel has the same depth as the input image" numbered="true" >}}

Convolution using a single filter:

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/3d-conv.png" title="Convolution using a single filter" numbered="true" >}}

Each filter actually happens to be a *collection of kernels*, with there being **one kernel for every single input channel** to the layer, and each kernel being unique. As the input image has 3 channels (RGB), our filter consists of also 3 kernels.

Each of the kernels of the filter “slides” over their respective input channels, producing a processed version of each.

![Image for post](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*8dx6nxpUh2JqvYWPadTwMQ.gif)

Each of the per-channel processed versions are then summed together to form *one* channel. The kernels of a filter each produce one version of each channel, and the filter as a whole produces one overall output channel.

![Image for post](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*CYB2dyR3EhFs1xNLK8ewiA.gif)

We can stack different filters to obtain a **multi-channel** output “image”.

For example, assuming that 

- input image has the size $\text{height} \times \text{width} \times \text{depth} = 32 \times 32 \times 3$

- filter size is $5 \times 5 \times 3$

- and we have 6 different $5 \times 5$ filters 

  $\to$ we’ll get 6 separate activation maps and stack it together

$\Rightarrow$ The depth of the multi-channel output "image" is 6. 

​	($depth\_\text{activation maps} = \\# filters$)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-19%2021.44.04.png" alt="截屏2020-08-19 21.44.04" style="zoom: 40%;" />

### Convolution Example

{{< figure src="https://ujwlkarn.files.wordpress.com/2016/08/giphy.gif?w=748" title="A filter (with red outline) slides over the input image (convolution operation) to produce a feature map. The convolution of another filter (with the green outline), over the same image gives a different feature map as shown. It is important to note that the Convolution operation captures the local dependencies in the original image. Also notice how these two different filters generate different feature maps from the same original image." numbered="true" >}}



### Non-linearity: ReLU 

For any kind of neural network to be powerful, it needs to contain non-linearity.  And CNN is no different.

After the convolution operation, we pass the result through **non-linear** activation function. In CNN we usually use **Rectified Linear Units (ReLU)**, because it has been [empirically observed](https://arxiv.org/pdf/1906.01975.pdf) that CNNs using ReLU are faster to train than their counterparts.
$$
\operatorname{ReLU}(x) = \max(0, x)
$$
<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/relu_graph.png" alt="relu graph" style="zoom: 25%;" />

### ReLU Example

{{< figure src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-6-18-19-pm.png?w=748" title="This example shows the ReLU operation applied to one of the fearure maps obtained in the convolutional example. The output feature map here is also referred to as the ‘Rectified’ feature map." numbered="true" >}}

### Stride and Padding

**Stride** specifies how much we move the convolution filter at each step. 

By default the value is 1:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*L4T6IXRalWoseBncjRr4wQ@2x.gif" alt="Image for post" style="zoom:40%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/d0ufdQE7LHA43cdSrVefw2I9DFceYMixqoZJ.gif" alt="img" style="zoom:50%;" />

Stride > 1often used to **down-sample** the image

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*4wZt9G7W7CchZO-5rVxl5g@2x.gif" alt="Image for post" style="zoom:40%;" />

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*BMngs93_rm2_BpJFH2mS0Q.gif" title="Stride 2 convolution" numbered="true" >}}

What do we do with border pixels?

$\to$ **Paddings**

- Fill up the image borders (zero-padding is most common)
- Preserve the size of the feature maps from shrinking
- Improves performance and makes sure the kernel and stride size will fit in the input

{{< figure src="https://miro.medium.com/max/700/1*W2D564Gkad9lj3_6t9I2PA@2x.gif" title="Height and width of feature map is same as the input image due to padding (the gray area)." numbered="true" >}}

### Dimension parameters computation

- Inpupt size: 

  $$W\_{1} \times H\_{1} \times D\_{1}$$

  (usually $W\_1 = H\_1$)

- Hyperparameters:

  - Number of filters: $K$
  - Filter size: $F \times F \times D\_1$
  - Stride: $S$
  - Amount of padding: $P$

- Output size:  
  $$
  W\_{2} \times H\_{2} \times K
  $$
  with

  - $W_{2}=\lfloor \frac{W_{1}-F+2 P}{S}+1 \rfloor$

  

  - $H_{2}=\lfloor \frac{H_{1}-F+2 P}{S}+1 \rfloor$

- Number of weights:
  $$
  \text{#weights} = \underbrace{F \cdot F}\_{\text {Filter size }} \cdot \underbrace{D\_{1}}_{\text {Filter depth }} \cdot \underbrace{K}\_{\text {#Filters }}
  $$
  

### Connections Calculation

$$
\text{#Connections} = \text{#Neurons of next layer} \times \text{filter size}
$$

Nice explanation from [cs231n](https://cs231n.github.io/convolutional-networks/#conv):

![截屏2020-08-27 10.09.10](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-27%2010.09.10.png)



### Summary of Conv-layer

1. Convolution operation using filters
2. Feed into ReLU

![007 CNN One Layer of A ConvNet | Master Data Science](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/new_NN_CNN_1_1.png)





## Pooling Layer

### How Pooling works?

After a convolution operation we usually perform **pooling** to reduce the dimensionality. 

Pooling layers downsample each feature map independently, reducing the height and width, keeping the depth intact. <span style="color:green">This enables us to reduce the number of parameters, which both shortens the training time and combats overfitting.</span> 👏

The most common type of pooling is **max pooling** which just takes the max value in the pooling window. Contrary to the convolution operation, pooling has NO parameters. It slides a window over its input, and simply takes the max value in the window. Similar to a convolution, we specify the window size and stride.

Example: max pooling using a $2 \times 2$ window and stride 2

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*ReZNSf_Yr7Q1nqegGirsMQ@2x.png" alt="Image for post" style="zoom: 40%;" />

Now let’s work out the feature map dimensions before and after pooling.

If the input to the pooling layer has the dimensionality $32 \times 32 \times 10$, using the same pooling parameters described above, the result will be a $16 \times 16 \times 10$ feature map.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*sExirX4-kgM0P66PysNQ4A@2x.png" alt="Image for post" style="zoom:40%;" />

Both the height and width of the feature map are *halved*. Thus we reduced the number of weights to 1/4 of the input.

The depth doesn’t change because pooling works independently on each depth slice the input.

{{% alert note %}}

In CNN architectures, pooling is typically performed with 2x2 windows, stride 2 and NO padding.

{{% /alert %}}

### Pooling Example

{{< figure src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-6-11-53-pm.png?w=748" title="This example shows the effect of Pooling on the Rectified Feature Map we received after the ReLU operation in the above ReLU example. Max refers to Max-Pooling and Sum refers to Sum-Pooling." numbered="true" >}}

### Why pooling works?

Because Pooling keeps the maximum value from each window, it **preserves the best fits of each feature within the window**. This means that it doesn’t care so much exactly where the feature fit as long as it fit somewhere within the window. 

The result of this is that CNNs can find whether a feature is in an image without worrying about where it is. This helps solve the problem of computers being hyper-literal.

In particular, Pooling 

- makes the input representations (feature dimension) smaller and more manageable
- reduces the number of parameters and computations in the network, therefore, controlling [overfitting](https://en.wikipedia.org/wiki/Overfitting) 
- makes the network invariant to small transformations, distortions and translations in the input image (a small distortion in input will not change the output of Pooling – since we take the maximum / average value in a local neighborhood).
- helps us arrive at an almost scale invariant representation of our image

### Dimension parameters computation

- Inpupt size: 

  $$W\_{1} \times H\_{1} \times D\_{1}$$

  (usually $W\_1 = H\_1$)

- Hyperparameters:

  - Number of filters: $K$
  - Filter size: $F \times F \times D\_1$
  - Stride: $S$
  - Typically no padding

- Output size:  
  $$
  W\_{2} \times H\_{2} \times D\_1
  $$
  with

  - $W_{2}=\lfloor \frac{W_{1}-F}{S}\rfloor+1 $

  

  - $H_{2}=\lfloor \frac{H_{1}-F}{S}\rfloor+1 $

- Number of weights: **0** (since it computes a fixed function of the input)

## Fully Connected Layer

After the convolution + pooling layers we add a couple of fully connected layers to wrap up the CNN architecture. 

![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Screen-Shot-2017-07-27-at-12.07.11-AM.png)

The Fully Connected layer is a traditional Multi Layer Perceptron that uses a softmax activation function in the output layer (other classifiers like SVM can also be used). The term “Fully Connected” implies that every neuron in the previous layer is connected to every neuron on the next layer.

The output from the convolutional and pooling layers represent high-level features of the input image. The purpose of the Fully Connected layer is to **use these features for classifying the input image into various classes based on the training dataset.**

> Remember that the output of both convolution and pooling layers are 3D volumes, but a fully connected layer expects a **1D vector of numbers**. So we *flatten* the output of the final pooling layer to a vector and that becomes the input to the fully connected layer. Flattening is simply arranging the 3D volume of numbers into a 1D vector, nothing fancy happens here.
>
> ![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Screen-Shot-2017-07-26-at-4.26.01-PM.png)

Apart from classification, adding a fully-connected layer is also a (usually) cheap way of learning non-linear combinations of these features. [Most of the features from convolutional and pooling layers may be good for the classification task, but combinations of those features might be even better](https://stats.stackexchange.com/questions/182102/what-do-the-fully-connected-layers-do-in-cnns/182122#182122).

## ✅ Advantages of CNN (vs. MLP)

- CNNs are good for translation invariance
- CNN reduces the numbers of parameters
  - Locally connected, shared weights, pooling, local feature extractor 
  - But learning power is still good or even better (generalization)
- We can “resize” the next layer **to as we want**
  - By setting kernel size, number of kernel, padding, stride
- Design of good architecture based on intuitions (or Neural architecture search)



## Reference

- [Intuitively Understanding Convolutions for Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
- [Applied Deep Learning - Part 4: Convolutional Neural Networks](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
- [Convolutional neural networks](https://www.jeremyjordan.me/convolutional-neural-networks/)

