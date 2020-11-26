---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 904

# Basic metadata
title: "Custom Datasets and Transforms"
date: 2020-11-26
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "PyTorch Recipe"]
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
        parent: pytorch-recipes
        weight: 4
---

## Custom Dataset

In order to use our custom dataset, we need to

- inherit `torch.utils.data.Dataset` , an abstract class representing a dataset.

- override
  - `__len__` so that `len(dataset)` returns the size of the dataset.
  - `__getitem__` to support the indexing such that `dataset[i]` can be used to get *i*-th sample.

The skeleton is as follows:

```python
from torch.utils.data.dataset import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, ...):
        # initial logic, e.g.
        # read csv
        # assign data transformation
        # ...
        
    def __getitem__(self, index):
        """Get the {index}-th sample"""
        # Note: the return value can be customized depending on application
        return (img, label)

    def __len__(self):
        return count # of how many examples(images?) you have
```

### Example

Let's take [MNIST](http://yann.lecun.com/exdb/mnist/) dataset as example. Assuming we have the csv file located in `CSV_PATH`. The structure of our csv file is

- One instance/sample per line
  - The first column is the digit label (0 - 9)
  - The rest 784 columns represents the values of each pixel in the image of size 28x28 ($28 \times 28 = 784$)
  - I.e. each sample consists of an image of digit and the label of the digit

- There're 5000 lines in total. I.e. 5000 samples
  - We want to use the first 4000 samples for training and validation, 
  - and the rest 1000 samples for testing.

Let's implement our custom MNIST dataset:

```python
from torch.utils.data import Dataset

class MyMNIST(Dataset):
    
    TRAIN, VALID, TEST = 0, 1, 2
    
    def __init__(self, csv_file, usage=TRAIN, transform=None, label_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file
            usage (int): usage of the dataset (train/validation/test)  
            transform (callable, optional): Optional transform to be applied on the image.
            label_transform (callable, optional): Optional transform to be applied on the label.
        """

        self.transform = transform # image preprocessing
        self.label_transform = label_transform # label preprocessing
        
        # load from csv file
        all_data = np.genfromtxt(csv_file, delimiter=',', dtype='uint8')
        
        # 5000 lines in csv file --> 5000 instances
        # training set: first 3000 lines
        # validation set: 3000 - 4000 
        # test set: last 1000 lines
        train, test = all_data[:4000], all_data[4000:]
        train, val = train[:3000], train[3000:]
        
        # choose lines based on specified usage
        if usage == self.TRAIN:
            self.images = train[:, 1:]
            self.labels = train[:, 0] # first column is label of the digit 
        elif usage == self.VALID:
            self.images = val[:, 1:]
            self.labels = val[:, 0]
        else:
            self.images = test[:, 1:]
            self.labels = test[:, 0]
      

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.label_transform is not None:
            label = self.label_transform(label)       
        # convert label to Tensor of dtype long
        label = torch.as_tensor(label, dtype=torch.long)
        
        return image, label
    
    
    def __len__(self):
        return len(self.labels)
             
```

Use our custom MNIST dataset:

```python
from torchvision import transforms

# apply normalizaton and convertion to Tensor before using the dataset
preprocess_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1,), (0.4))])

# let's say we use the dataset for testing
my_mnist = MyMNIST(csv_file=CSV_PATH,
                  usage=MyMNIST.TEST,
                  transform=preprocess_transform)
```

## Custom transform and augmentation

The example code above takes use of the transforms provided by `torchvision.transforms`. We can also implement custom transforms by ourselves. 

To do this, we need to write them as **callable** classes:

- inherit `object` class
- implement `__init___` if needed
- define desired transformations in `__call__(self, image)` method

### Example

For example, let's implement two custom transforms:

```python
class MyNormalizer(object):
    """Normalize image"""

    def __call__(self, image):
        """
        Only works for our custom MNIST dataset: Devide the pixel values by 255
        Generally, normalization should work as follows:
        data_normalized = (data - data.mean) / data.std
        """
        image = image * 1.0 / 255
        return image


class MyToTensor(object):
    """Convert image to PyTorch Tensor"""
 
    def __call__(self, image):
        image = torch.from_numpy(image).float()
        return image
```

### Use custom transform in our custom MNIST dataset

```python
preprocess_transform = transforms.Compose([MyToTensor(),
                                          MyNormalizer()])

my_mnist = MyMNIST(csv_file=CSV_PATH,
                  usage=MyMNIST.TEST,
                  transform=preprocess_transform)
```



## Reference

- [WRITING CUSTOM DATASETS, DATALOADERS AND TRANSFORMS](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms)

