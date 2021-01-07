---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 170

# Basic metadata
title: "📈 Visualization with TensorBoard"
date: 2020-09-22
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch"]
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
        parent: getting-started
        weight: 7
---

## TL;DR

0. Define network structure, loss function, optimizer

1. Set up TensorBoard

   1. Import `SummaryWriter` from `torch.utils.tensorboard`

   2. Define `SummaryWriter`

   
```python
   from torch.utils.tensorboard import SummaryWriter
   
   # let's say we'll log for "fashion_mnist_experiment_1"
   NAME = "fashion_mnist_experiment_1"
   
   # default `log_dir` is "./runs"
   writer = SummaryWriter(f'runs/{LOG_DIR}')
   ```
   
2. Launch TensorBoard

   - If we launch in Jupyter Notebook/Lab or Google Colab, we need to load the TensorBoard notebook extension first

   ```python
   %load_ext tensorboard
   %tensorboard --logdir=runs
   ```

   - If we run on terminal, execute the command

   ```python
   tensorboard --logdir=runs
   ```

   Then go to the URL it provides OR to http://localhost:6006/

3. Inspect the model 

   - using `add_graph()` 

   - remember to call `writer.close()`

   

   ```python
   # assume that net is our neural network
   # and images are a batch of training images
   
   writer.add_graph(net, images)
   writer.close()
   ```

4. Track model training with `add_scalar()`

5. Access trained models

   - Precision-Recall curve: `add_pr_curve()`




## Useful functions

### `add_scalar()`[^1]

- Add scalar data to summary.

  - Scalar helps to save the loss value of each training step, or the accuracy after each epoch.

- Lots of information can be logged for one experiment. To avoid cluttering the UI and have better result clustering, we can group plots by naming them *hierarchically*. 

  - For example, “Loss/train” and “Loss/test” will be grouped together.

  ```python
  writer.add_scalar('Loss/train', np.random.random(), n_iter)
      writer.add_scalar('Loss/test', np.random.random(), n_iter)
      writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
      writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
  ```

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/hier_tags.png" alt="_images/hier_tags.png" style="zoom:80%;" />

### `add_scalars()` [^2]

- Adds many scalar data in the same plot to summary. 

- Use case: compare train loss and validation loss to see if it's overfitting

- Example:

  ```python
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter()
  r = 5
  for i in range(100):
      writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                      'xcosx':i*np.cos(i/r),
                                      'tanx': np.tan(i/r)}, i)
  writer.close()
  ```

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/add_scalars.png" alt="_images/add_scalars.png" style="zoom:50%;" />



## Google Colab Notebook

- [Colab Notebook](https://colab.research.google.com/drive/16Qsog13Ehj6CWFb4q5VP5GFoKfeoFIlb?authuser=1)
- [tensor board.ipynb](https://colab.research.google.com/drive/1U_htRSmqZFAPKwZ3xoMWixvWHfxPYXPL#scrollTo=kv59lFDZYbwR)

## Resource

- [tensorboardX documentation](https://tensorboardx.readthedocs.io/en/latest/index.html)
- [HOW TO USE TENSORBOARD WITH PYTORCH](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
- Pytorch documentation: [TORCH.UTILS.TENSORBOARD](https://pytorch.org/docs/stable/tensorboard.html)

## Reference

- [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#)

[^1]: https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar
[^2]: https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalars