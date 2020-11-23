---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 305

# Basic metadata
title: "Numpy Random"
date: 2020-11-23
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Numpy"]
categories: ["coding"]
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
    python:
        parent: numpy
        weight: 5
---

## Common used functions

```python
import numpy as np
```

We use `numpy.random` package to generate an array of random values instead of looping through the random generation of one variable.

### `np.random.randn()`

- Return a sample (or samples) from the “standard normal” distribution.

- [Documentation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html)

- Example

  ```python
  np.random.randn(2,4)
  ```

  ```
  [[-0.50516164  0.03107622 -1.98470915 -0.06278207]
   [ 0.00806484  1.60814316 -0.06865081  0.90962803]]
  ```

### `np.random.rand()`

- Create an array of the given shape and populate it with random samples from a uniform distribution over `[0, 1)`.

- [Documentation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html)

- Example

  ```python
  np.random.rand(2,4)
  ```

  ```
  [[0.93885544 0.0643788  0.74463388 0.97446713]
   [0.03621414 0.0420926  0.54597933 0.72757245]]
  ```

### `np.random.randint()`

- Return random integers from the “discrete uniform” distribution in the “half-open” interval [*low*, *high*). If *high* is None (the default), then results are from [0, *low*).

- [Documentation](https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.randint.html)

- Example

  ```python
  np.random.randint(0, 10, size=(2, 3))
  ```

  ```
  array([[2, 6, 0],
         [1, 5, 8]])
  ```

### `np.random.permutation`()

- Randomly permute a sequence, or return a permuted range.

- [Documentation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html)

- Example

  ```python
  np.random.permutation([1, -4, 3, 2, -6])
  ```

  ```
  [-6  2  1  3 -4]
  ```

### `np.random.seed()`

The random algorithm used for all of the methods above is a ***pseudo random generating algorithm***. It is based on some initial state, or "*seed*" to generate random numbers. 

- If you do not specify the seed, it can take some number elsewhere to be the seed. 
- But if we specify the same seed every time we generate random numbers, those numbers will be the same. It is good for many cases, for example, replicating exactly the result of your neural network training even you randomly initialized your weights.

Example:

```python
print("When we do not specify the seed:")
for i in range(3):
    print(np.random.randint(0,10))

print("When we specify the same seed:")
for i in range(3):
    np.random.seed(1111) # set the same seed 1111 before every random generation
    print(np.random.randint(0,10))
    
print("When we specify different seeds:")
for i in range(3):
    np.random.seed(i * 3)
    print(np.random.randint(0,10))
```

```
When we do not specify the seed:
9
5
5
When we specify the same seed:
7
7
7
When we specify different seeds:
5
8
9
```

## Random initialization

For a neural network to learn well, beside feature normalization and other things, we also need proper weight initialization: 

- the weights should be randomly initialized, 
- or at least different numbers (to break the symmetry), 
- and they should be small. 

There're two good methods for good initialization. They both take the size of the layers into account (and to be more precisely, they also consider the activation functions).

### Xavier initialization

It's good when use with `sigmoid` or `tanh` activation functions.


$$
\alpha = \displaystyle\sqrt{\frac{2}{n_{in} + n_{out}}}
$$

- $n\_{in}$: the number of neurons from the previous layer
- $n\_{out}$: the number of neurons from the current layer

```python
n_in = 3 # the number of neurons from the previous layer
n_out = 2 # the number of neurons from the current layer
alpha = np.sqrt(2. / (n_in + n_out))
W = np.random.randn(n_out, n_in) * alpha
print(W)
```

```
[[-0.64820888  1.28386704  0.08342666]
 [ 1.0910627   1.16949752  0.63448025]]
```

{{% alert note %}} 

Further reading: [An Explanation of Xavier Initialization](https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)

{{% /alert %}}

Summary:

![initializations](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/initializations.png)

### He initialization

It's good when use with `ReLU` activation function.
$$
\alpha = \displaystyle\sqrt{\frac{1}{n_{in}}}
$$


##  Sampling

`np.random.choice()`

- Generates a random sample from a given 1-D array

- [Documentation](https://docs.scipy.org/doc//numpy-1.10.4/reference/generated/numpy.random.choice.html)

- Example

  ```python
  def sm_sample_general(out, smp):
      # out contains possible outputs
      # smp contains the softmax output distributions
      return np.random.choice(out, p = smp)
  
  out = ['a', 'b', 'c']
  smp=np.array([0.3, 0.6, 0.1])
  
  outputs = []
  for i in range(10):
      outputs.append(sm_sample_general(out, smp))
  print(outputs)
  
  outputs = []
  # Law of large numbers: 100000 is large enough for our sample to approximate the true distribution
  for i in range(100000):  
      outputs.append(sm_sample_general(out, smp))
  
  from collections import Counter
  c_list = Counter(outputs)
  print(c_list) 
  ```

  ```
  ['c', 'b', 'b', 'a', 'c', 'a', 'b', 'b', 'a', 'b']
  Counter({'b': 60044, 'a': 29928, 'c': 10028})
  ```

## Dropout

- Act as a regularization, aimming to make the network less prone to overfitting.

- In training phase, with Dropout, at each hidden layer, with probability `p`, we **kill** the neuron. 

  - What it means by ‘kill’ is to set the neuron to 0. As neural net is a collection multiplicative operations, then those 0 neuron won’t propagate anything to the rest of the network.

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/dropout.png" style="width:679px;height:321px">

  - Let `n` be the number of neuron in a hidden layer, then the expectation of the number of neuron to be active at each Dropout is `p*n`, as we sample the neurons uniformly with probability `p`. 

    - Concretely, if we have 1024 neurons in hidden layer, if we set `p = 0.5`, then we can expect that only half of the neurons (512) would be active at each given time.
    - Because we force the network to train with only random `p*n` of neurons, then intuitively, we force it to learn the data with different kind of neurons subset. The only way the network could perform the best is to adapt to that constraint, and learn the more general representation of the data.

- Implmentation

  1. Sample an array of independent Bernoulli Distribution, which is just a collection of zero or one to indicate whether we kill the neuron or not.

     - If we multiply our hidden layer with this array, what we get is the originial value of the neuron if the array element is 1, and 0 if the array element is also 0.

  2. Scale the layer output with `p`

     - Necause we’re only using `p*n` of the neurons, the output then has the expectation of `p*x`, if `x` is the expected output if we use all the neurons (without Dropout).

       As we don’t use Dropout in test time, then the expected output of the layer is `x`. That doesn’t match with the training phase. What we need to do is to make it matches the training phase expectation, so we need to scale the output with `p`

  ```python
  # Dropout training, notice the scaling of 1/p
  u1 = np.random.binomial(1, p, size=h1.shape) / p
  h1 *= u1
  ```

  

{{% alert note %}} 

Further reading: [Implementing Dropout in Neural Net](https://wiseodd.github.io/techblog/2016/06/25/dropout/)

{{% /alert %}}





