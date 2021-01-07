---
# Title, summary, and position in the list
linktitle: "YOLOv3 (2)"
summary: ""
weight: 502

# Basic metadata
title: "YOLOv3 (2): Creating the layers of the network"
date: 2020-11-09
draft: true
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
        weight: 2
---

## Getting started

- Create a directory for the code
- Then, create a file `darknet.py`. 
  - **Darknet is the name of the underlying architecture of YOLO**. 
  - This file will contain the code that creates the YOLO network. We will supplement it with a file called `util.py` which will contain the code for various helper functions.

## Configuration file

The official code (authored in C) uses a configuration file (`.cfg` file) to build the network. The `.cfg` file describes the layout of the network, block by block. 

Download the official *cfg* gile and place it in a folder called `cfg`

```bash
mkdir cfg
cd cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo
```

The content of the *cfg* file looks like this:

```
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear
...
```

{{< figure src="https://europepmc.org/articles/PMC6695703/bin/sensors-19-03371-g010.jpg" title="YOLO v3 structure" numbered="true" >}}

There're **5** types of layers that are used in YOLO:

- **Convolutional**
- **Shortcut**
- **Upsample**
- **Route**
- **YOLO**

### Convolutional

Example:

```
[convolutional]
batch_normalize=1  
filters=64  
size=3  
stride=1  
pad=1  
activation=leaky
```

### Shortcut

Example:

```
[shortcut]
from=-3  
activation=linear  
```

- A *shortcut* layer is a skip connection, similar to the one used in ResNet. 
- `from=-3` means: the output of the shortcut layer is obtained by **adding** feature maps from the previous and the **3rd layer backwards** from the *shortcut* layer.

### Upsample

Example:

```
[upsample]
stride=2
```

Upsamples the feature map in the previous layer by a factor of `stride` using bilinear upsampling.

### Route

Example:

```
[route]
layers = -4

[route]
layers = -1, 61
```

-  It has an attribute `layers` which can have either one, or two values.
  - When `layers` attribute has only **one** value, it outputs the feature maps of the layer indexed by the value. 
    - In our example, it is **-4**, so the layer will output feature map from the **4th layer backwards** from the *Route*layer.
  - When `layers` has **two** values, it returns the concatenated feature maps of the layers indexed by it's values.
    - In our example it is -1 and 61, and the layer will output feature maps **from the previous layer (-1) and the 61st layer, concatenated along the depth dimension**.

> More see: https://github.com/AlexeyAB/darknet/issues/487#issuecomment-374902735

### YOLO

```
[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
```

- YOLO layer corresponds to the **Detection** layer
  - The `anchors` describes 9 anchors, but only the anchors which are indexed by attributes of the `mask` tag are used.
    - Here, the value of `mask` is 0,1,2, which means the first, second and third anchors are used.

### Information about the network: **Net** 

```
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
width= 320
height = 320
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
```

Another type of block called `net` in the *cfg*

- only describes information about the network input and training parameters.
- isn't used in the forward pass of YOLO. 

## Parsing the configuration file

The idea here is to parse the *cfg*, and **store every block as a dict**. The attributes of the blocks and their values are stored as key-value pairs in the dictionary. 

As we parse through the *cfg*, we keep appending these dicts, denoted by the variable `block` in our code, to a list `blocks`.

```python
def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    with open(cfgfile, 'r') as f:
      	# store the lines in a list
        lines = f.read().split('\n') 
        # remove empty lines and comments
        lines = [x for x in lines if len(x) > 0 and x[0] != '#'] 
        # remove fringe whitespaces
        lines = [x.strip() for x in lines] 

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[': # marks the start of a new block
            if len(block) != 0:
                # if block is not empty,
                # implies it is storing values of previous block
                # then add the previous block to blocks list
                blocks.append(block)
                # re-init the block for current new block
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
        
    blocks.append(block)
    
    return blocks
```

## Creating the building blocks

The idea is to iterate over the list of blocks returned by the `parse_cfg` function, and create a PyTorch module for each block.

Before we start, there're some points need to be noticed:

- `nn.ModuleList`

  Our function will return a `nn.ModuleList`. This class is almost like a normal list containing `nn.Module` objects.  when we add `nn.ModuleList` as a member of a `nn.Module` object (i.e. when we add modules to our network), all the `parameter`s of `nn.Module` objects (modules) inside the `nn.ModuleList` are added as `parameter`s of the `nn.Module` object.

- Dimension of convolutional layer's kernel
  - When we define a new convolutional layer, we must define the dimension of it's kernel. While the height and width of kernel is provided by the *cfg* file, **the depth of the kernel is precisely the number of filters (or depth of the feature map) present in the previous layer**. 
  - This means we need to **keep track of number of filters in the layer on which the convolutional layer is being applied**. 
  - We use the variable `prev_filter` to do this. We initialise this to 3, as the image has 3 filters corresponding to the RGB channels.
- The route layer brings (possibly concatenated) feature maps from previous layers. If there's a convolutional layer right in front of a route layer, then the kernel is applied on the feature maps of previous layers, precisely the ones the route layer brings. Therefore, we need to keep a track of the number of filters in not only the previous layer, but **each** sone of the preceding layers. As we iterate, we append the number of output filters of each block to the list `output_filters`.

### `convolutional` layer

We use `nn.Sequential` to sequentially execute a number of `nn.Module` objects.

- A block may contain more than one layer. 
- We string together these layers using the `nn.Sequential` and the `add_module` function.

```python
if x["type"] == "convolutional":
    # get the information about the layer
    activation = x["activation"]
    try:
        batch_normalize = int(x["batch_normalize"])
        bias = False
    except:
        batch_normalize = 0
        bias = True

    filters = int(x["filters"])
    padding = int(x["pad"])
    kernel_size = int(x["size"])
    stride = int(x["stride"])

    pad = (kernel_size - 1) // 2 if padding else 0

    # Add the CONV layer
    conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
    module.add_module(f"conv_{index}", conv)

    # Add the Batch Norm layer
    if batch_normalize:
        bn = nn.BatchNorm2d(filters)
        module.add_module(f"batch_norm_{index}", bn)

    # activation function
    if activation == "leaky":
        activn = nn.LeakyReLU(0.1, inplace=True)
        module.add_module(f"leaky_{index}", activn)
```

### `upsample` layer

```python
elif x["type"] == "upsample":
    stride = int(x["stride"])
    upsample = nn.Upsample(scale_factor=2, mode="nearest")
    module.add_module(f"upsample_{index}", upsample)
```

### `route` layer

```python
elif x["type"] == "route":
    x["layers"] = x["layers"].split(',')

    # start of a route
    start = int(x["layers"][0]) 

    # end of a route, if it exists
    try:
        end = int(x["layers"][1])
    except:
        end = 0

    # positive anotaiton
    if start > 0:
        start = start - index
    if end > 0:
        end = end - index

    route = EmptyLayer()
    module.add_module(f"route_{index}", route)
		
    # updates the filters variable to hold the number of filters outputted by a route layer.
    if end < 0:
      	# if we're concatenating maps
        filters = output_filters[index + start] + output_filters[index + end]
    else:
        filters = output_filters[index + start]
```

Here we'll define a new layer called `EmptyLayer` in advance:

```python
class EmptyLayer(nn.Module):
  	"""
  	Just an empty layer
  	"""
    def __init__(self):
        super(EmptyLayer, self).__init__()
```

> Why define a dummy `EmptyLayer`?
>
> As we mentioned, the route layer brings (possibly concatenated) feature maps from previous layers. Civen the code of concatenation is fairly short and simple (calling `torch.cat` on feature maps), designing a layer as above will thus lead to unnecessary abstraction that just increases boiler plate code. Instead, what we can do is put a dummy layer in place of a proposed route layer, and then perform the concatenation directly in the `forward` function of the `nn.Module` object representing darknet.

### `shortcut` layer

The `shortcut` layer performs a very simple operation (addition). There is no need to update update the `filters` variable as it merely adds a feature maps of a previous layer to those of layer just behind.

```python
elif x["type"] == "shortcut":
    shortcut = EmptyLayer()
    module.add_module(f"shortcut_{index}", shortcut)
```

### YOLO Layer

We define a new layer `DetectionLayer` that holds the anchors used to detect bounding boxes.

```python
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
```

```python
elif x["type"] == "yolo":
    mask = x["mask"].split(",")
    mask = [int(x) for x in mask]

    anchors = x["anchors"].split(",")
    anchors = [int(a) for a in anchors]
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask]

    detection = DetectionLayer(anchors)
    module.add_module(f"Detection_{index}", detection)
```

### Putting them together 

```python
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        
def create_modules(blocks):
  	"""
  	Takes a blocks list according to the parsing of cfg file,
  	create PyTorch modules for the blocks
  	"""
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # 1. check the type of the block
        # 2. create a new module for the block
        # 3. append to module_list

        if x["type"] == "convolutional":
            # get the information about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            pad = (kernel_size - 1) // 2 if padding else 0

            # Add the CONV layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f"conv_{index}", conv)

            # Add the Batch Norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)

            # activation function
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{index}", activn)

        elif x["type"] == "upsample":
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module(f"upsample_{index}", upsample)

        elif x["type"] == "route":
            x["layers"] = x["layers"].split(',')

            # start of a route
            start = int(x["layers"][0]) 

            # end of a route, if it exists
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # positive anotaiton
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module(f"route_{index}", route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut coreesponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{index}", shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{index}", detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)
```

## Testing the code

```python
blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))
```

we will see a long list, (exactly containing 106 items), the elements of which will look like

```
...
(9): Sequential(
  (conv_9): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (batch_norm_9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (leaky_9): LeakyReLU(negative_slope=0.1, inplace=True)
)
(10): Sequential(
  (conv_10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (batch_norm_10): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (leaky_10): LeakyReLU(negative_slope=0.1, inplace=True)
)
(11): Sequential(
	(shortcut_11): EmptyLayer()
)
...
```

