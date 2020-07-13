# Kernel Methods

## Kernel function

Given a mapping function $\phi: \mathcal{X} \rightarrow \mathcal{V}$, the function 

$$
\mathcal{K}: x \rightarrow v, \quad \mathcal{K}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\left\langle\phi(\mathbf{x}), \phi\left(\mathbf{x}^{\prime}\right)\right\rangle_{\mathcal{V}}
$$

is called a **kernel function**.

*"A kernel is a function that returns the result of a dot product performed in another space."*



## Kernel trick

Applying the kernel trick simply means **replacing the dot product of two examples by a kernel function**. 

### Typical kernels

| Kernel Type                                       | Definition                                                   |
| ------------------------------------------------- | ------------------------------------------------------------ |
| **Linear kernel**                                 | $k\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\left\langle\boldsymbol{x}, \boldsymbol{x}^{\prime}\right\rangle$ |
| **Polynomial kernel**                             | $k\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\left\langle\boldsymbol{x}, \boldsymbol{x}^{\prime}\right\rangle^{d}$ |
| **Gaussian / Radial Basis Function (RBF) kernel** | $k \left(\boldsymbol{x}, \boldsymbol{y}\right)=\exp \left(-\frac{\|\boldsymbol{x}-\boldsymbol{y}\|^{2}}{2 \sigma^{2}}\right)$ |

### Why do we need kernel trick?

- Kernels can be used for all feature based algorithms that can be rewritten such that they contain **inner products** of feature vectors
  - This is true for almost all feature based algorithms (Linear regression, SVMs, ...)

- Kernels can be used to map the data $\mathbf{x}$ in an infinite dimensional feature space (i.e., a function space)
  - **The feature vector never has to be represented explicitly**
  - **As long as we can evaluate the inner product of two feature vectors**

➡️ We can obtain a more powerful representation than standard linear feature models.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/Kernel_trick.png" alt="Kernel_trick" style="zoom:80%;" />

