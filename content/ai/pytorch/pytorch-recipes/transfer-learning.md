---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 901

# Basic metadata
title: "Transfer Learning for Computer Vision"
date: 2020-11-03
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
        parent: pytorch-recipes
        weight: 1
---



## Handling settings for training and valiadtion phase flexibly

💡 Use Python dictionary

- Phase (`'train'` or `'val'`) as key

For example:

```python
data_transforms = {
    # For training: data augmentation and normalization
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    # For validation: only normalization
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'hymenoptera_data'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                          data_transforms[x]) 
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, 
                                              shuffle=True, num_workers=4) 
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
```



## General function to train a model

Here we will

- schedule the learning rate
- save hte best model

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    scheduler is an LR scheduler object from torch.optim.lr_scheduler
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}')
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the params gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only in trianing phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * inputs.shape[0]
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```

## Major Transfer Learning scenarios 

In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. **Instead, it is common to pretrain a ConvNet on a very large dataset** (e.g. ImageNet, which contains 1.2 million images with 1000 categories), **and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.** 

### ConvNet as fixed feature extractor

1. Take a ConvNet pretrained on ImageNet
2. Remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet)
3. Treat the rest of the ConvNet as a fixed feature extractor for the new dataset. (We call these features **CNN codes**.)

#### Implementation with PyTorch

- we will freeze the weights for all of the network except that of the final fully connected layer. 
- This last fully connected layer is replaced with a new one with random weights and **only this layer is trained**.

```python
# Load pretrained model
model_conv = torchvision.models.resnet18(pretrained=True)

# Freeze all the network
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
# in other words, now we freeze all the network except the final layer
num_features = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_features, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

Train and evaluate:

```python
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

### Fine-tuning the ConvNet

The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to **also fine-tune the weights of the pretrained network by continuing the backpropagation**. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. 

Motivation: the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. 

#### Implementation with PyTorch

- Instead of random initializaion, we initialize the network with a pretrained network, like the one that is trained on imagenet 1000 dataset. 
- Rest of the training looks as usual.

```python
# Load a pretrained model
model_ft = models.resnet18(pretrained=True)

# Reset the final fully connected layer according to specific task
num_features = model_ft.fc.in_features
num_classes = 2 # assuming a binary classification task
model_ft.fc = nn.Linear(num_features, num_classes) 

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay learning rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

Train and evaluate:

```python
model_conv = train_model(model_ft, criterion=criterion, optimizer=optimizer_ft,
                       scheduler=exp_lr_scheduler, num_epochs=25)
```

## When and how to fine-tune?

The two most important factors are:

- size of the new dataset (small or big)
- its similarity to the original dataset 

Keeping in mind that **ConvNet features are more generic in early layers and more original-dataset-specific in later layers.**

Common rules of thumb for navigating the 4 major scenarios:

1. ***New dataset is small and similar to original dataset*.** 

   Since the data is small, it is not a good idea to fine-tune the ConvNet due to overfitting concerns. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.

2. ***New dataset is large and similar to the original dataset*.** 

   Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.

3. ***New dataset is small but very different from the original dataset*.** 

   Since the data is small, it is likely best to only train a linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network.

4. ***New dataset is large and very different from the original dataset*.** 

   Since the dataset is very large, we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.

### Pratical advices

- ***Constraints from pretrained models*.** 
  - Note that if you wish to use a pretrained network, you may be slightly constrained in terms of the architecture you can use for your new dataset. For example, you can’t arbitrarily take out Conv layers from the pretrained network. 
  - However, some changes are straight-forward: Due to parameter sharing, you can easily run a pretrained network on images of different spatial size. This is clearly evident in the case of Conv/Pool layers because their forward function is *independent* of the input volume spatial size (as long as the strides “fit”). 
  - In case of FC layers, this still holds true because FC layers can be converted to a Convolutional Layer: For example, in an AlexNet, the final pooling volume before the first FC layer is of size [6x6x512]. Therefore, the FC layer looking at this volume is equivalent to having a Convolutional Layer that has receptive field size 6x6, and is applied with padding of 0.
- ***Learning rates*.** 
  - It’s common to use a **smaller** learning rate for ConvNet weights that are being fine-tuned, in comparison to the (randomly-initialized) weights for the new linear classifier that computes the class scores of your new dataset. 
  - This is because we expect that the ConvNet weights are relatively good, so we don’t wish to distort them too quickly and too much (especially while the new Linear Classifier above them is being trained from random initialization).



















## Reference

- [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#)

- [CS231n-Transfer Learning](https://cs231n.github.io/transfer-learning/)

