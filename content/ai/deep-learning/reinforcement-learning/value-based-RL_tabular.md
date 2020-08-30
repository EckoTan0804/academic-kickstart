---
# Title, summary, and position in the list
linktitle: "Value-based RL (Tabular) "
summary: ""
weight: 830

# Basic metadata
title: "Value-based RL : The Tabular Case"
date: 2020-08-24
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Optimization", "Reinforcement Learning"]
categories: ["Deep Learning"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?

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
    deep-learning:
        parent: rl
        weight: 3

---

**Optimal Control =** Given an MDP $(S, A, P, R, \gamma, H)$, find the **optimal policy $\pi^\*$**



| Exact methods                                                | Limitations                                                  | Solutions                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------- |
| Value Iteration                                              | Updates require dynamics model                               | Sampling-based approximations |
| **Policy Iteration**  <li>Policy evaluation <li>Policy improvement | Iteration over all states + actions requires small discrete problems | Function approximation        |

## Sampling-based Policy Evaluation

**We want to estimate the value function**
$$
V^{\pi}(s)=\mathbb{E}\_{\pi}\left[\sum\_{t=0}^{\infty} \gamma^{t} r\_{t} \mid s\_{0}=s\right]
$$

- Expected future reward if agent is in state and follows policy

- By using sample trajectories 
  $$
  \boldsymbol{\tau}=\left(\boldsymbol{s}\_{0}, \boldsymbol{a}\_{0}, r\_{0}, \boldsymbol{s}\_{1}, \boldsymbol{a}\_{1}, r\_{1}, \ldots\right)
  $$
  (no model $P(s’|s,a)$ available)

**2 Options**:

- Monte Carlo (MC) Estimates
- Temporal Difference Learning

### Monte Carlo Estimates

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2015.42.40.png" alt="截屏2020-08-24 15.42.40" style="zoom:60%;" />

**Limitations:**

- Returns are very noisy!
- Sample inefficient!
- We have to wait until episode is over 🤪

### Sample-based policy evaluation by dynamic programming

- **Policy evaluation for current policy $\pi$**
  $$
  \begin{aligned}
  V\_{k}^{\pi}(s) &=\sum\_{a} \pi(\boldsymbol{a} \mid s)\left(r(s, \boldsymbol{a})+\gamma \sum\_{\boldsymbol{s}^{\prime}} P\left(\boldsymbol{s}^{\prime} \mid \boldsymbol{s}, \boldsymbol{a}\right) V\_{k-1}^{\pi}\left(\boldsymbol{s}^{\prime}\right)\right) \\\\
  &=\mathbb{E}\_{\pi}\left[r(\boldsymbol{s}, \boldsymbol{a})+\gamma \mathbb{E}\_{P}\left[V\_{k-1}^{\pi}\left(\boldsymbol{s}^{\prime}\right)\right]\right]
  \end{aligned}
  $$

- **Expectations can be approximated by sampling!**

  - Sample action:
    $$
    \boldsymbol{a}\_{t} \sim \pi\_{i}\left(\cdot \mid s\_{t}\right)
    $$

  - Sample next state:
    $$
    \boldsymbol{s}\_{t+1} \sim P\left(\cdot \mid \boldsymbol{s}\_{t}, \boldsymbol{a}\_{t}\right)
    $$

  - Compute new target for current time step $t$:
    $$
    y\_{t}=r\left(s\_{t}, \boldsymbol{a}\_{t}\right)+\gamma V^{\pi}\left(\boldsymbol{s}\_{t+1}\right)
    $$

  - Incorporate target value by moving average:
    $$
    V^{\pi}\left(s\_{t}\right) \leftarrow(1-\alpha) V^{\pi}\left(s\_{t}\right)+\alpha y\_{t}
    $$

    - $\alpha$: learning rate

### Temporal Difference (TD) learning

The resulting algorithm:
$$
\begin{aligned}
V^{\pi}\left(s\_{t}\right) & \leftarrow(1-\alpha) V^{\pi}\left(s\_{t}\right)+\alpha\left(r\left(s\_{t}, a\_{t}\right)+\gamma V^{\pi}\left(s\_{t+1}\right)\right) \\\\
&=V^{\pi}\left(s\_{t}\right)+\alpha(\underbrace{r\left(s\_{t}, a\_{t}\right)+\gamma V^{\pi}\left(s\_{t+1}\right)-V^{\pi}\left(s\_{t}\right)}\_{\delta\_{t}})
\end{aligned}
$$

- $\delta\_t$: the temporal difference

**Can be seen as combination of MC and Dynamic Programming**

- Only 1 transition is used to generate the targets... 
- ... as opposed to many transitions in MC

- Less noisy, more sample efficient :clap:

### Algorithm

![截屏2020-08-24 16.07.16](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2016.07.16.png)

### Q-Learning

Sample-based Q-Value Iteration:

![截屏2020-08-24 16.11.27](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2016.11.27.png)

**We do not know the real Q-values:**

- By using the greedy policy, we might miss out better actions where the value is still wrong
- So we still **need to explore!**
- **Exploration-Exploitation Tradeoff**, one of the hardest problems in RL

**Common Exploration Policies (discrete actions)**

- **Epsilon-Greedy Policy:** Take random action with probability $\epsilon$
  $$
  \pi(\boldsymbol{a} \mid s)=\begin{cases}
  1-\epsilon+\epsilon /|\mathcal{A}|, \text { if } \boldsymbol{a}=\operatorname{argmax}\_{\boldsymbol{a}^{\prime}} Q^{\pi}\left(\boldsymbol{s}, \boldsymbol{a}^{\prime}\right) \\\\
  \epsilon / \mid \mathcal{A}, \text { otherwise }
  \end{cases}
  $$

- **Soft-Max Policy:** Higher Q-Value, higher probability.
  $$
  \pi(a \mid s)=\frac{\exp (Q(s, a) / \beta)}{\sum_{a^{\prime}} \exp \left(Q\left(s, a^{\prime}\right) / \beta\right)}
  $$

  - $\beta$: temperature

#### (Tabular) Q-learning Algorithm

![截屏2020-08-24 16.15.26](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2016.15.26.png)

#### Proterties

- **Amazing result:** Q-learning converges to optimal policy -- even if you’re acting suboptimally! :clap:

- **Caveats:**

  - You have to explore enough
    - **In theory:** visit every state action pair infinitely often

  - You have to eventually make the learning rate small enough 
    - ... but not decrease it too quickly

- **Technical requirements:**

  - All states and actions are visited infinitely often

    - Basically, in the limit, it doesn’t matter how you select actions (!)

  - Learning rate schedule such that for all state and action pairs $(s,a)$
    $$
    \begin{aligned}
    \sum\_{t=0}^{\infty} \alpha\_{t}(s, a) &= \infty \\\\
    \sum\_{t=0}^{\infty} \alpha\_{t}(s, a)^2 &< \infty
    \end{aligned}
    $$

    - we need a decreasing learning rate, e.g: $\alpha = \frac{1}{t}$
    - In practice, often a **constant** learning rate is chosen