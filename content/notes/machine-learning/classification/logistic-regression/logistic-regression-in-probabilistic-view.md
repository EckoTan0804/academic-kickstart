# Logistic Regression (Probabilistic view)

Class label: 

$
y_i \in \{0, 1\}
$

Conditional probability distribution of the class label is

$
\begin{align}
p(y=1|\boldsymbol{x}) &= \sigma(\boldsymbol{w}^T\boldsymbol{x}+b) \\
p(y=0|\boldsymbol{x}) &= 1 - \sigma(\boldsymbol{w}^T\boldsymbol{x}+b)
\end{align}
$

with 

$\sigma(x) = \frac{1}{1+\operatorname{exp}(-x)}$

This is a **conditional Bernoulli distribution**. Therefore, the probability can be represented as

$
\begin{array}{ll}
p(y|\boldsymbol{x}) &= p(y=1|\boldsymbol{x})^y p(y=0|\boldsymbol{x})^{1-y} \\
& = \sigma(\boldsymbol{w}^T\boldsymbol{x}+b)^y (1 - \sigma(\boldsymbol{w}^T\boldsymbol{x}+b))^{1-y}
\end{array}
$

The **conditional Bernoulli log-likelihood** is (assuming training data is i.i.d)

$
\begin{align}
\operatorname{loglik}(\boldsymbol{w}, \mathcal{D}) 
&= \log(\operatorname{lik}(\boldsymbol{w}, \mathcal{D})) \\
&= \log(\displaystyle\prod_i p(y_i|\boldsymbol{x}_i)) \\
&= \log\left(\displaystyle\prod_i \sigma(\boldsymbol{w}^T\boldsymbol{x}_i+b)^y \left(1 - \sigma(\boldsymbol{w}^T\boldsymbol{x}_i+b)\right)^{1-y}\right) \\
&= \displaystyle\sum_i y\log\left(\sigma(\boldsymbol{w}^T\boldsymbol{x}_i+b)\right)+ (1-y)\log\left(1 - \sigma(\boldsymbol{w}^T\boldsymbol{x}_i+b)\right) 
\end{align}
$

Let 

$
\tilde{\boldsymbol{w}}=\left(\begin{array}{c}1 \\ \boldsymbol{w} \end{array}\right), \quad \tilde{\boldsymbol{x}_i}=\left(\begin{array}{c}b \\ \boldsymbol{x}_i \end{array}\right)
$

Then:

$
\operatorname{loglik}(\boldsymbol{w}, \mathcal{D}) = \operatorname{loglik}(\tilde{\boldsymbol{w}}, \mathcal{D})  = \displaystyle\sum_i y\log\left(\sigma(\tilde{\boldsymbol{w}}^T\tilde{\boldsymbol{x}_i})\right)+ (1-y)\log\left(1 - \sigma(\tilde{\boldsymbol{w}}^T\tilde{\boldsymbol{x}_i}))\right)
$

Our objective is to find the $\tilde{\boldsymbol{w}}^*$ that **maximize the log-likelihood**, i.e.

$
\begin{array}{cl}
\tilde{\boldsymbol{w}}^* &= \underset{\tilde{\boldsymbol{w}}}{\arg \max} \quad \operatorname{loglik}(\tilde{\boldsymbol{w}}, \mathcal{D}) \\
&= \underset{\tilde{\boldsymbol{w}}}{\arg \min} \quad -\operatorname{loglik}(\tilde{\boldsymbol{w}}, \mathcal{D})\\
&= \underset{\tilde{\boldsymbol{w}}}{\arg \min} \quad \underbrace{-\left(\displaystyle\sum_i y\log\left(\sigma(\tilde{\boldsymbol{w}}^T\tilde{\boldsymbol{x}_i})\right) + (1-y)\log\left(1 - \sigma(\tilde{\boldsymbol{w}}^T\tilde{\boldsymbol{x}_i}))\right)\right)}_{\text{cross-entropy loss}}
\end{array}
$

In other words, **maximizing the (log-)likelihood is the same as minimizing the cross entropy.**