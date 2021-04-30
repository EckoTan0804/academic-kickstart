---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 202

# Basic metadata
title: "Semantic Segmentation with PyTorch"
date: 2021-23-29
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Segmentation", "Semantic Segmentation"]
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
        parent: segmentation
        weight: 2
---

## What is Semantic Segmentation?

Semantic Segmentation is an image analysis task in which we classify each pixel in the image into a class.

Let's say we have the following image:

![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/index3.png)

Its semantically segmentated image would be the following:

![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/index4.png)

Each pixel in the image is classified to its respective class.

## Use PyTorch for Semantic Segmentation

### Input and Output

- Segmentation models expect a 3-channled image which is normalized with the Imagenet mean and standard deviation, i.e.,
  `mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]`.

- Input is `[Ni x Ci x Hi x Wi]`
  - `Ni` -> the batch size
  - `Ci` -> the number of channels (which is 3)
  - `Hi` -> the height of the image
  - `Wi` -> the width of the image

- Output of the model is `[No x Co x Ho x Wo]`
  - `No` -> is the batch size (same as `Ni`)
  - `Co` -> **is the number of classes that the dataset have!**
  - `Ho` -> the height of the image (which is the same as `Hi` in almost all cases)
  - `Wo` -> the width of the image (which is the same as `Wi` in almost all cases)
- The `torchvision` models outputs an `OrderedDict` and not a `torch.Tensor`.
  And in `.eval()` mode it just has one key `out`. The `out` key of this `OrderedDict` is the key that holds the output and this `out` key's value has the shape of `[No x Co x Ho x Wo]`.

### Implementation

```python
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import models
import torchvision.transforms as T

# ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# # Pascal VOC dataset segmentation
VocClass = namedtuple("VocClass", ["name", "id", "color"])
classes = [
    VocClass("background", 0, (0, 0, 0)),
    VocClass("aeroplane", 1, (128, 0, 0)),
    VocClass("bicycle", 2, (0, 128, 0)),
    VocClass("bird", 3, (128, 128, 0)),
    VocClass("boat", 4, (0, 0, 128)),
    VocClass("bottle", 5, (128, 0, 128)),
    VocClass("bus", 6, (0, 128, 128)),
    VocClass("car", 7, (128, 128, 128)),
    VocClass("cat", 8, (64, 0, 0)),
    VocClass("chair", 9, (192, 0, 0)),
    VocClass("cow", 10, (64, 128, 0)),
    VocClass("dining table", 11, (192, 128, 0)),
    VocClass("dog", 12, (64, 0, 128)),
    VocClass("horse", 13, (192, 0, 128)),
    VocClass("motorbike", 14, (64, 128, 128)),
    VocClass("person", 15, (192, 128, 128)),
    VocClass("potted plant", 16, (0, 64, 0)),
    VocClass("sheep", 17, (128, 64, 0)),
    VocClass("sofa", 18, (0, 192, 0)),
    VocClass("train", 19, (128, 192, 0)),
    VocClass("tv/monitor", 10, (0, 64, 128)),
]


def decode_seg_map(image):
    """
    Convert a segmentation map of size [1 x num_class x H x W] to a 2D RGB image
    """
    
    # Create empty 2D matrices for all 3 channels of an image
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for class_ in classes:
        # Get the indexes in the image where that particular class label is present
        idx = (image == class_.id)

        # Put its corresponding color to those pixels
        r[idx], g[idx], b[idx] = class_.color

    # Stack the 3 seperate channels to form a RGB image
    rgb_mask = np.stack([r, g, b], axis=2)
    return rgb_mask


def show_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    

def segment(model, img_path, show_original_img=True, device="cuda"):
    img = Image.open(img_path)
    if show_original_img:
        show_img(img)
    
    transform = T.Compose([
        T.Resize(640),
        # T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

    input = transform(img).unsqueeze(0).to(device)
    output = model.to(device)(input)["out"]
    seg_map = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    mask = decode_seg_map(seg_map)
    show_img(mask)
```

```python
!wget -nv "https://www.learnopencv.com/wp-content/uploads/2021/01/person-segmentation.jpeg" -O person.png
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
segment(fcn, "./person.png")
```

![person](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/person.png)

![person_sem_seg](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/person_sem_seg.png)

## Reference

[intro_seg.ipynb](https://colab.research.google.com/github/spmallick/learnopencv/blob/master/PyTorch-Segmentation-torchvision/intro-seg.ipynb#scrollTo=5GA_GNohUHnR&uniqifier=1)

