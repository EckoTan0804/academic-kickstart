---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 908

# Basic metadata
title: "TorchScript"
date: 2021-04-21
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch", "PyTorch Recipe", "TorchScript"]
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
        weight: 8
---

![截屏2021-04-21 17.17.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2021-04-21%2017.17.23.png)



## TorchScript

- A PyTorch model’s journey from Python to C++ is enabled by **Torch Script**, a representation of a PyTorch model that can be understood, compiled and serialized by the Torch Script compiler.
- Any TorchScript program can be saved from a Python process and loaded in a process where there is NO Python dependency. In other words, a TorchScript program can be run **independently** from Python, such as in a standalone C++ program.
- This makes it possible to train models in PyTorch using familiar tools in Python and then export the model via TorchScript to a production environment where Python programs may be disadvantageous for performance and multi-threading reasons.
- 👍 Advantage
  - TorchScript code can be invoked in its own interpreter, which is basically a restricted Python interpreter. This interpreter does not acquire the Global Interpreter Lock, and so many requests can be processed on the same instance simultaneously.
  - This format allows us to save the whole model to disk and load it into another environment, such as in a server written in a language other than Python
  - TorchScript gives us a representation in which we can do compiler optimizations on the code to provide more efficient execution
  - TorchScript allows us to interface with many backend/device runtimes that require a broader view of the program than individual operators.

### Steps for Loading a PyTorch Model in C++

1. Converte PyTorch Model to TorchScript
2. Serialize script module to a file
3. Load script module in C++
4. Execute script module in C++

## Convert PyTorch Model to Torch Script

There are wo ways to convert a PyTorch model to Torch Script

- [Tracing](#tracing)
- [Scripting](#scripting)

### Tracing

- A mechanism in which
  - the structure of the model is captured by evaluating it once using example inputs and 
  - recording the flow of those inputs through the model.
- Suitable for models that make limited use of control flow
- Function: `torch.jit.trace`

#### Example

```python
import torch

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
```

What happens under the hood when we call `torch.jit.trace`, passing in the `Module` and an example input?

- It has invoked the `Module`
- Recorded the operations that occured when the `Module `was run
- Created an instance of `torch.jit.ScriptModule`

TorchScript records its definitions in an **Intermediate Representation** (or **IR**), commonly referred to in Deep learning as a *graph* (we can examine the graph with the `.graph` property). 

A better way is to use the `.code` property to give a Python-syntax interpretation of the code:

```python
print(traced_cell.code)
```

 Out:

```txt
def forward(self,
    input: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  _0 = torch.add((self.linear).forward(input, ), h, alpha=1)
  _1 = torch.tanh(_0)
  return (_1, _1)
```

### Scripting

If our code use control flows (if-else, loop...), then tracing is unsuitable. In this case, we will use a **script compiler**, which does code analysis of our Python source code to transform it into TorchScript. The function for compiling the module is `torch.jit.script`.

#### Example

```python
import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output
    
my_module = MyModule(10,20)
sm = torch.jit.script(my_module)
```

`sm` is an instance of `ScriptModule` that is ready for serialization.

### Mixing Scripting and Tracing

In many cases either tracing or scripting is an easier approach for converting a model to TorchScript. Tracing and scripting can be composed to suit the particular requirements of a part of a model.

**Scripted functions can call traced functions.**

- Useful when we need to use control-flow around a simple feed-forward model

- Example

  ```python
  import torch
  
  def foo(x, y):
      return 2 * x + y
  
  traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
  
  @torch.jit.script
  def bar(x):
      return traced_foo(x, x)
  ```

**Traced functions can call script functions.**

- Useful when a small part of a model requires some control-flow even though most of the model is just a feed-forward network.

- Control-flow inside of a script function called by a traced function is preserved correctly.

- Example

  ```python
  import torch
  
  @torch.jit.script
  def foo(x, y):
      if x.max() > y.max():
          r = x
      else:
          r = y
      return r
  
  
  def bar(x, y, z):
      return foo(x, y) + z
  
  traced_bar = torch.jit.trace(bar, (torch.rand(3), torch.rand(3), torch.rand(3)))	
  ```

  

## Saving aand Loading Script Module

- Save: `save()`
- Load: `torch.jit.load()`

Example: 

```python
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
```

- Save:

  ```python
  traced_script_module.save("traced_resnet_model.pt")
  ```

- Load:

  ```python
  traced_resnet = torch.jit.load("traced_resnet_model.pt")
  ```

  



## Reference

- [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#basics-of-torchscript)

- [Loading a TorchScript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html#step-2-serializing-your-script-module-to-a-file)

- [Torch Script](https://pytorch.org/docs/stable/jit.html#)



