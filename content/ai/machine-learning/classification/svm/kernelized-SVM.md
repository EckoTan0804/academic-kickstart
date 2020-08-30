---
# Basic info
title: "Kernelized SVM"
date: 2020-07-13
draft: false
# type: docs # page type
authors: ["admin"]
tags: ["ML", "Classification"]
categories: ["Machine Learning"]
toc: true # Show table of contents?
weight: 30

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: "SVM with kernel tricks" 
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
---

## SVM (with features)

- Maximum margin principle

- Slack variables allow for margin violation
  $$
  \begin{array}{ll} \underset{\mathbf{w}}{\operatorname{argmin}} \quad &\|\mathbf{w}\|^{2} + C \sum_i^N \xi_i \\\\ \text { s.t. } \quad & y_{i}\left(\mathbf{w}^{T} \color{red}{\phi(\mathbf{x}_{i})} + b\right) \geq 1 -\xi_i, \quad \xi_i \geq 0\end{array}
  $$



## Math basics

Solve the constrained optimization problem: **Method of Lagrangian Multipliers**

- **Primal optimization problem**:

$$
\begin{array}{ll}
\underset{\boldsymbol{x}}{\min} \quad & f(\boldsymbol{x}) \\\\
\text { s.t. } \quad & h_{i}(\boldsymbol{x}) \geq b_{i}, \text { for } i=1 \ldots K
\end{array}
$$

- **Lagrangian optimization**:

$$
\begin{array}{ll}
\underset{\boldsymbol{x}}{\min} \underset{\boldsymbol{\lambda}}{\max} \quad & L(\boldsymbol{x}, \boldsymbol{\lambda}) = f(\boldsymbol{x}) - \sum_{i=1}^K \lambda_i(h_i(\boldsymbol{x}) - b_i) \\\\
\text{ s.t. } &\lambda_i\geq 0,  \quad i = 1\dots K
\end{array}
$$

- **Dual optimization problem**
  $$
  \begin{aligned}
  \boldsymbol{\lambda}^{\*}=\underset{\boldsymbol{\lambda}}{\arg \max } g(\boldsymbol{\lambda}), \quad & g(\boldsymbol{\lambda})=\min \_{\boldsymbol{x}} L(\boldsymbol{x}, \boldsymbol{\lambda}) \\\\
  \text { s.t. } \quad \lambda_{i} \geq 0, & \text { for } i=1 \ldots K
  \end{aligned}
  $$

  - $g$ : **dual function** of the optimization problem
  - Essentially swapped min and max in the definition of $L$

- **Slaters condition:** For a **convex** objective and **convex** constraints, **solving the dual is equivalent to solving the primal**

  - I.e., optimal primal parameters can be obtained from optimal dual parameters
    $$
    \boldsymbol{x}^* = \underset{\boldsymbol{x}}{\operatorname{argmin}}L(\boldsymbol{x}, \boldsymbol{\lambda}^*)
    $$



## Dual derivation of the SVM

1. SVM optimization:
   $$
   \begin{array}{ll}
   &\underset{\boldsymbol{w}}{\operatorname{argmin}} \quad &\|\boldsymbol{w}\|^2 \\
   &\text{ s.t. } \quad &y_i(\boldsymbol{w}^T\phi(\mathbf{x}_i) + b) \geq 1
   \end{array}
   $$

2. Lagrangian function:
   $$
   L(\boldsymbol{w}, \boldsymbol{\lambda})=\frac{1}{2} \boldsymbol{w}^{T} \boldsymbol{w}-\sum_{i} \alpha_{i}\left(y_{i}\left(\boldsymbol{w}^{T} \phi\left(\boldsymbol{x}_{i}\right)+b\right)-1\right)
   $$

3. Compute optimal $\boldsymbol{w}$
   $$
   \begin{align}
   &\frac{\partial L}{\partial \boldsymbol{w}} = \boldsymbol{w} - \sum_i \alpha_i y_i \phi(\boldsymbol{x}_i) \overset{!}{=} 0 \\\\
   \Leftrightarrow \quad & \color{CornflowerBlue}{\boldsymbol{w}^* = \sum_i \alpha_i y_i \phi(\boldsymbol{x}_i)}
   \end{align}
   $$

   - Many of the $\alpha_i$  will be zero (constraint satisfied)

   - If $\alpha_i \neq 0 \overset{\text{complementary slackness}}{\Rightarrow}  y_{i}\left(\boldsymbol{w}^{T} \phi\left(\boldsymbol{x}_{i}\right)+b\right)-1 =0$ 

     $\Rightarrow \phi(\boldsymbol{x}_i)$ is a support vector  

   - The optimal weight vector $\boldsymbol{w}$ is a **linear combination of the support vectors**! 👏

4. Optimality condition for $b$:
   $$
   \frac{\partial L}{\partial b} = - \sum_i \alpha_i y_i  \overset{!}{=} 0 \quad \Rightarrow \sum_i \alpha_i y_i = 0
   $$

   - We do not obtain a solution for $b$
   - But an additional condition for $\alpha$

   $b$ can be computed from $w$: 

   If  $\alpha\_i > 0$, then $\boldsymbol{x}\_i$ is on the margin due to complementary slackness condition. I.e.: 
   $$
   \begin{align}y_{i}\left(\boldsymbol{w}^{T} \phi\left(\boldsymbol{x}_{i}\right)+b\right)-1 &= 0 \\\\y_{i}\left(\boldsymbol{w}^{T} \phi\left(\boldsymbol{x}_{i}\right)+b\right) &= 1 \\\\ \underbrace{y_{i} y_{i}}_{=1}\left(\boldsymbol{w}^{T} \phi\left(\boldsymbol{x}_{i}\right)+b\right) &= y_{i} \\\\ \Rightarrow b = y_{i} - \boldsymbol{w}^{T} \phi\left(\boldsymbol{x}_{i}\right)\end{align}
   $$



## Apply kernel tricks for SVM

- Lagrangian: 

$$
L(\boldsymbol{w}, \boldsymbol{\lambda}) = {\color{red}{\frac{1}{2} \boldsymbol{w}^{T} \boldsymbol{w}}} - \sum_{i} \alpha\_{i}\left({\color{green}{y\_{i} (w^{T} \phi\left(x_{i}\right)}}+ b)-\color{CornflowerBlue}{1}\right), \quad \boldsymbol{w}^{\*}=\sum\_{i} \alpha_{i} y\_{i} \phi\left(\boldsymbol{x}\_{i}\right)
$$

- Dual function (**Wolfe Dual Lagrangian function**):

$$
\begin{aligned}
g(\boldsymbol{\alpha}) &=L\left(\boldsymbol{w}^{*}, \boldsymbol{\alpha}\right) \\\\
&=\color{red}{\frac{1}{2} \underbrace{\sum_{i} \sum_{j} \alpha_{i} \alpha_{j} y_{i} y_{j} \phi\left(\boldsymbol{x}_{i}\right)^{T} \phi\left(\boldsymbol{x}_{j}\right)}_{{\boldsymbol{w}^*}^T \boldsymbol{w}^*}} - \color{green}{\sum_{i} \alpha_{i} y_{i}(\underbrace{\sum_{j} \alpha_{j} y_{j} \phi\left(x_{j}\right)}_{\boldsymbol{w}^*})^{T} \phi\left(x_{i}\right)} + \color{CornflowerBlue}{\sum_{i} \alpha_{i}} \\\\
&=\sum_{i} \alpha_{i}-\frac{1}{2} \sum_{i} \sum_{j} \alpha_{i} \alpha_{j} y_{i} y_{j} \underbrace{\phi\left(\boldsymbol{x}_{i}\right)^{T} \phi\left(\boldsymbol{x}_{j}\right)}_{\overset{}{=} \boldsymbol{k}(\boldsymbol{x}_i, \boldsymbol{x}_j)} \\\\
&= \sum_{i} \alpha_{i}-\frac{1}{2} \sum_{i} \sum_{j} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{k}(\boldsymbol{x}_i, \boldsymbol{x}_j )
\end{aligned}
$$

- **Wolfe dual optimization problem**:

$$
\begin{array}{ll}
\underset{\boldsymbol{\alpha}}{\min} \quad & \sum_{i} \alpha_{i}-\frac{1}{2} \sum_{i} \sum_{j} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{k}(\boldsymbol{x}_i, \boldsymbol{x}_j ) \\\\
\text{ s.t } \quad & \alpha_i \geq 0 \quad \forall i = 1, \dots, N \\\\
& \sum_i \alpha_i y_i = 0
\end{array}
$$

- **Compute primal from dual parameters**:

  - **Weight vector** 
    $$
    \boldsymbol{w}^{*}=\sum_{i} \alpha_{i} y_{i} \phi\left(\boldsymbol{x}_{i}\right)
    \label{eq:weight vector}
    $$

    - Can not be represented (as it is potentially infinite dimensional). But don't worry, we don't need the explicit representation

  - **Bias**: For any $i$ with $\alpha_i > 0$ : 

  $$
  \begin{array}{ll}
  b &=y_{k}-\mathbf{w}^{T} \phi\left(\boldsymbol{x}_{k}\right) \\\\
  &=y_{k}-\sum_{i} y_{i} \alpha_{i} k\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{k}\right)
  \end{array}
  $$

  - **Decision function** (Again, we use the kernel trick and therefore we don't need the explicit representation of the weight vector $\boldsymbol{w}^*$)

  $$
  \begin{aligned}f(\boldsymbol{x}) &= (\boldsymbol{w}^{*})^{T} \boldsymbol{\phi}(\boldsymbol{x}) + b \\\\
  &\overset{}{=} \left(\sum_{i} \alpha_{i} y_{i} \phi\left(\boldsymbol{x}_{i}\right)\right)^{T}  \boldsymbol{\phi}(\boldsymbol{x}) + b \\\\
  &= \sum_{i} \alpha_{i} y_{i} \boldsymbol{\phi}(\boldsymbol{x}_i)^{T} \boldsymbol{\phi}(\boldsymbol{x}) + b \\\\
  & \overset{}{=}\sum_i y_{i} \alpha_{i} k\left(\boldsymbol{x}_{i}, \boldsymbol{x}\right)+b\end{aligned}
  $$



## Relaxed constraints with slack variable

- **Primal optimization problem**
  $$
  \begin{array}{ll} \underset{\mathbf{w}}{\operatorname{argmin}} \quad &\|\mathbf{w}\|^{2} + \color{CornflowerBlue}{C \sum_i^N \xi_i} \\\\
  \text { s.t. } \quad & y_{i}\left(\mathbf{w}^{T} \mathbf{x}_{i} + b\right) \geq 1 - \color{CornflowerBlue}{\xi_i}, \quad \color{CornflowerBlue}{\xi_i} \geq 0\end{array}
$$
  
- **Dual optimization problem**
  $$
  \begin{array}{ll}\underset{\boldsymbol{\alpha}}{\min} \quad & \sum_{i} \alpha_{i}-\frac{1}{2} \sum_{i} \sum_{j} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{k}(\boldsymbol{x}_i, \boldsymbol{x}_j ) \\\\ \text{ s.t } \quad & \color{CornflowerBlue}{C \geq} \alpha_i \geq 0 \quad \forall i = 1, \dots, N \\\\ & \sum_i \alpha_i y_i = 0\end{array}
  $$
  
    <span style="color:CornflowerBlue">Add upper bound of </span> $\color{CornflowerBlue}{C}$ <span style="color:CornflowerBlue">on</span> $\color{CornflowerBlue}{\alpha_i}$

  - Without slack, $\alpha_i \to \infty$ when constraints are violated (points misclassified)
  - Upper bound of $C$ limits the $\alpha_i$, so misclassifications are allowed

