---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 1004

# Basic metadata
title: "Training Issues"
date: 2021-02-23
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "NN Training"]
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
        parent: training
        weight: 4
---

Here I will document some problems and issues I encountered while training the neural network.

## Validation accuracy higher than training accuracy

Possible reasons 

- **Regularization**

  Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned on at training time and turned off at testing time. 

  - Dropout

    When training, a percentage of the features are set to zero. When testing, all features are used (and are scaled appropriately). So the model at test time is more robust - and can lead to higher testing accuracies.

    > Ref: [Higher validation accuracy, than training accurracy using Tensorflow and Keras](https://stackoverflow.com/questions/43979449/higher-validation-accuracy-than-training-accurracy-using-tensorflow-and-keras)

- **Data augmentation**

  Applying data augmentation to the training data makes the task significantly harder for the neural network. 

  > Ref: [Why is validation accuracy higher than training accuracy when applying data augmentation?](https://stackoverflow.com/questions/48845354/why-is-validation-accuracy-higher-than-training-accuracy-when-applying-data-augm)
  
- **Dataset is small**

  **Smaller datasets have smaller intrinsic variance**. This means that the model properly captures patterns inside of your data and train error is greater simply because the inner variance of training set is greater then validation set.

  > Ref:[Keras cifar10 example validation and test loss lower than training loss](https://stackoverflow.com/questions/42878683/keras-cifar10-example-validation-and-test-loss-lower-than-training-loss)

## CUDA error (59): Device-side assert triggered

The error occurs due to the following two reasons:

1. Inconsistency between the number of labels/classes and the number of output units
2. The input of the loss function may be incorrect.

### Inconsistency between the number of labels/classes and the number of output units

I came across this error when I was working on the [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist). 

In this dataset, every image represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions). In other words, the greatest label is 24 and labels are **noncontinuous** (0, 1, ..., 8, 10, 11, ..., 24). There're totally 24 label classes.

So, I just naively designed the last FC layer as followings:

```python
self.fc3 = nn.Linear(48, 24)
```

Then this error occurred. Why?

The error is usually identified in the line where you do the backpropagation. Your loss function will be comparing the *output* from your model and the *label* of that observation in your dataset. In my case, output dimension of the last FC layer is 24, meaning that the greatest possible value for class label prediction is 23 (counting from zero). However, in this dataset, some of the labels have value 24, which is beyond the range (24 > 23)! This causes the error to be triggered! 

How to fix it?

***Make sure the number of output units match the number of your classes***

In my case, I changed the output dimension of the last FC layer from 24 to 25: 

```python
self.fc3 = nn.Linear(48, 25)
```

Then everything works well!

### Wrong input for the loss function

Loss functions have different ranges for the possible inputs that they can accept. If you choose an incompatible activation function for your output layer this error will be triggered. For example, BCELoss requires its input to be between 0 and 1. If the input(output from your model) is beyond the acceptable range for that particular loss function, the error gets triggered.

### Small extra tip

The error messages you get when running into this error may not be very descriptive. To make sure you get the complete and *useful* stack trace, have this at the very beginning of your code and run it before anything else:

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
```

> Reference: 
>
> - [CUDA error 59: Device-side assert triggered](https://towardsdatascience.com/cuda-error-device-side-assert-triggered-c6ae1c8fa4c3)
> - [Debugging CUDA device-side assert in PyTorch](https://lernapparat.de/debug-device-assert/)

