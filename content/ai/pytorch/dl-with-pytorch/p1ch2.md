---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 202

# Basic metadata
title: "Pretrained Networks"
date: 2020-10-18
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "DL-with-PyTorch"]
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
        parent: dl-with-pytorch
        weight: 2
---

## Pretrained Network for Object Recognition

### Use pretrained network in `TorchVision`

The [**TorchVision** project](https://github.com/pytorch/vision) 

- contains a few of the best-performing neural network architectures for computer vision, such as
  - AlexNet (http://mng.bz/lo6z)
  - ResNet (https://arxiv.org/pdf/1512.03385.pdf)
  - Inception v3 (https://arxiv.org/pdf/1512.00567.pdf)
- has easy access to datasets like ImageNet and other utilities for getting up to speed with computer vision applications in PyTorch.

The predefined models can be found in `torchvision.models`

```python
from torchvision import models

dir(models)
```

```
['AlexNet',
 'DenseNet',
 'GoogLeNet',
 'GoogLeNetOutputs',
 'Inception3',
 'InceptionOutputs',
 'MNASNet',
 'MobileNetV2',
 'ResNet',
 'ShuffleNetV2',
 'SqueezeNet',
 'VGG',
 '_GoogLeNetOutputs',
 '_InceptionOutputs',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 '_utils',
 'alexnet',
 'densenet',
 'densenet121',
 'densenet161',
 'densenet169',
 'densenet201',
 'detection',
 'googlenet',
 'inception',
 'inception_v3',
 'mnasnet',
 'mnasnet0_5',
 'mnasnet0_75',
 'mnasnet1_0',
 'mnasnet1_3',
 'mobilenet',
 'mobilenet_v2',
 'quantization',
 'resnet',
 'resnet101',
 'resnet152',
 'resnet18',
 'resnet34',
 'resnet50',
 'resnext101_32x8d',
 'resnext50_32x4d',
 'segmentation',
 'shufflenet_v2_x0_5',
 'shufflenet_v2_x1_0',
 'shufflenet_v2_x1_5',
 'shufflenet_v2_x2_0',
 'shufflenetv2',
 'squeezenet',
 'squeezenet1_0',
 'squeezenet1_1',
 'utils',
 'vgg',
 'vgg11',
 'vgg11_bn',
 'vgg13',
 'vgg13_bn',
 'vgg16',
 'vgg16_bn',
 'vgg19',
 'vgg19_bn',
 'video',
 'wide_resnet101_2',
 'wide_resnet50_2']
```

- The **capitalized names** (e.g. ResNet) refer to Python classes that implement a number of popular models. They differ in their architecture—that is, in the arrangement of the operations occurring between the input and the output.

  - E.g.: create an instance of the `AlexNet` class.

    ```python
    # create an instance of AlexNet class
    alexnet = models.AlexNet()
    ```

    But wait! If we did that, we would be feeding data through the whole network to produce ... garbage!!! :cry:

    **That’s because the network is uninitialized: its weights, the numbers by which inputs are added and multiplied, have not been trained on anything—the network itself is a blank (or rather, random) slate.** We’d need to either train it from scratch or load weights from prior training. 

    To use models with predefined numbers of layers and units and optionally download and load pretrained weights into them, we need to use the **lowercase name** in `models` module.

- The **lowercase names** are convenience functions that return models instantiated from those classes, sometimes with different parameter sets.

  - For instance, `resnet101` returns an instance of ResNet with 101 layers, `resnet18` has 18 layers, and so on.

  - Create an instance of the network and pass an argument that will instruct the function to download the weights of `resnet101` trained on the ImageNet dataset, with 1.2 million images and 1,000 categories:

    ```python
    resnet = models.resnet101(pretrained=True)
    ```

### Load and show an image from the local filesystem

Use Pillow (https://pillow.readthedocs.io/en/stable), an image-manipulation module for Python:

```python
from PIL import Image

# assume that the variable IMG_PATH holds the path of the image
img = Image.open(IMG_PATH)
img # show the image inline
```

### Set `eval` mode before inference

In order to do inference, we need to put the network in `eval` mode:

```python
resnet.eval()
```

*(If we forget to do that, some pretrained models, like batch normalization and dropout, will not produce meaningful answers, just because of the way they work internally.)*

### Retrieve image label 

1. **load a text file listing the labels in the same order they were presented to the network during training**
2. **Pick out the label at the index that produced the highest score from the network.**

(Almost all models meant for image recognition have output in a form similar to that)



## Torch Hub

Torch Hub is **a mechanism through which authors can publish a model on GitHub, with or without pretrained weights, and expose it through an interface that PyTorch understands.** This makes loading a pretrained model from a third party as easy as loading a TorchVision model.

All it takes is to place a file named **hubconf.py** in the root directory of the GitHub repository. An example is [TorchVision](https://github.com/pytorch/vision), we can notice that it contains a **hubconf.py**.

Torch Hub is quite new, and there are only a few models published this way. We can get at them by Googling “github.com hubconf.py.” 

