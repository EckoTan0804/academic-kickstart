---
# Title, summary, and position in the list
linktitle: "MDPs"
summary: ""
weight: 820

# Basic metadata
title: "Markov Decision Process (MDP)"
date: 2020-08-24
draft: true
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
        weight: 2

---

## "Easy" case

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2000.13.45.png" alt="截屏2020-08-24 00.13.45" style="zoom:33%;" />

**Markov Decision Process:**

- $\boldsymbol{s}$: states
- $\boldsymbol{a}$: actions
- $r(\boldsymbol{s}, \boldsymbol{a})$: reward model
- $p(\boldsymbol{s}'| \boldsymbol{s}, \boldsymbol{a})$: transition model
- $\mu\_0(\boldsymbol{s})$: initial state distribution

**Markov Property: Transitions only depend on current time step**
$$
p\left(\boldsymbol{s}\_{t+1} | \boldsymbol{s}\_{t}, \boldsymbol{a}\_{t}, \boldsymbol{s}\_{t-1}, \boldsymbol{a}\_{t-1}, \boldsymbol{s}\_{t-2}, \ldots\right)=p\left(\boldsymbol{s}\_{t+1} | \boldsymbol{s}\_{t}, \boldsymbol{a}\_{t}\right)
$$
🎯 Goal: 
$$
\arg \max \_{\pi} J\_{\pi}=\mathbb{E}\_{\pi}\left[\sum\_{t=0}^{\infty} \gamma^{t} r\_{t}\right]
$$
**How would we do this if all the models**

- transition model
- reward model 

**are known?**

### Optimal Quantities

- **The (optimal) valueof a state $s$:**

  $V^*(s) = $ expected summed reward starting in $s$ and acting optimally

- **The (optimal) value of a state action pair $(s,a)$:**

  $Q^*(s, a)=$ expected summed reward starting out having taken action $a$ from state $s$ and (thereafter) acting optimally

- **The optimal policy:**

  $\pi(s)=$ optimal action from state $s$

## Solving MDPs: Value Iteration 

> “An optimal sequence of controls in a multistage optimization problem has the property that **whatever the initial stage, state and controls are**, the **remaining controls** must constitute **an optimal sequence of decisions for the remaining problem** with stage and state resulting from previous controls considered as initial conditions.”
>
> -- Richard Bellman, Dynamic Programming, 1957

💡 **In order to achieve the optimal value at state $s$, we also must act optimally at the successor states** :muscle:

- **Act optimally at state $s$ (maximization step)**
  $$
  V^{\*}(s)=\max \_{a} Q^{\*}(s, a)
  $$

- **Q-value assumes we also act optimally at successor state**
  $$
  Q^{\*}(s, a)=r(s, a)+\gamma \sum\_{s^{\prime}} P\left(s^{\prime} \mid s, a\right) V^{*}\left(s^{\prime}\right)
  $$

$\to$ **Bellman Optimality Principle**: *Recursive* definition of optimal values
$$
V^{\*}(s)=\max \_{a}\left(r(s, a)+\gamma \sum\_{s^{\prime}} P\left(s^{\prime} \mid s, a\right) V^{*}\left(s^{\prime}\right)\right)
$$
**Time-Limited Values:** **Define $V\_k(s)$ to be the optimal value of $s$ if the MDP ends in $k$ more time steps**

- This can be computed efficiently from $V\_{k-1}(s)$
  $$
  V\_{k}^{\*}(s)=\max \_{a}\left(r(s, a)+\gamma \sum\_{s^{\prime}} P\left(s^{\prime} \mid s, a\right) V\_{k-1}^{*}\left(s^{\prime}\right)\right)
  $$

- Like a **1-step expectimax** with terminal values $V\_{k-1}(s)$

### Algorithm

![截屏2020-08-24 10.25.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2010.25.23.png)

### Example

![截屏2020-08-24 10.59.52](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2010.59.52.png)

E.g., if the car is <span style="color:blue">cool</span>, and it goes <span style="color:red">fast</span>

- 50% chance stay <span style="color:blue">cool</span>, and 50% chance become <span style="color:brown">warm</span>
- <span style="color:green">Reward is +2</span>

{{< figure src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2011.24.37.png" title="Value calculation in details" numbered="true" >}}

### Retrieval of the optimal policy

- **Infinite Horizon**
  $$
  \pi^{\*}(s)=\underset{a}{\arg \max }\left(r(s, a)+\gamma \sum\_{s^{\prime}} P\left(s^{\prime} | s, a\right) V^{\*}\left(s^{\prime}\right)\right)
  $$

  - **Note:** the infinite horizon optimal policy is **stationary**, i.e., the optimal action at state s is the **same** action at all times. (Efficient to store!)

- **Finite Horizon**
  $$
  \pi_{t}^{\*}(s)=\underset{a}{\arg \max }\left(r(s, a)+\sum_{s^{\prime}} P\left(s^{\prime} | s, a\right) V_{H-t}^{\*}\left(s^{\prime}\right)\right)
  $$

  - Optimal policy is **time dependent**
  - Always iterate until horizon $H$, need to store the value functions for each iteration $k$

### Convergence

**Theorem: Value iteration converges** :clap:

 At convergence, we have found the optimal value function $V^*$ for the discounted infinite horizon problem, which satisfies the Bellman
$$
\forall s \in S: V^{\*}(s)=\max \_{a}\left(r(s, a)+\gamma \sum\_{s^{\prime}} P\left(s^{\prime} | s, a\right) V^{\*}\left(s^{\prime}\right)\right)
$$

### Q-Values

**The (optimal) value of a state action pair $(s,a)$:**
$$
Q^{\*}(s, a)=\left(r(s, a)+\gamma \sum\_{s^{\prime}} P\left(s^{\prime} | s, a\right) \max \_{a^{\prime}} Q^{\*}\left(s^{\prime}, a^{\prime}\right)\right)
$$

- $Q^{\*}(s, a)$: expected summed reward starting out having taken action a from state s and (thereafter) acting optimally

**We can also do value iteration with the Q-values**:
$$
Q_{k}^{\*}(s, a)=\left(r(s, a)+\gamma \sum\_{s^{\prime}} P\left(s^{\prime} | s, a\right) \max \_{a^{\prime}} Q_{k-1}^{\*}\left(s^{\prime}, a^{\prime}\right)\right)
$$
**Policy can be directly obtained from Q:**
$$
\pi^{\*}(s)=\arg \max \_{a} Q^{\*}(s, a)
$$

## Solving MDPs: Policy Iteration

**policy iteration**: Policy-centric approach for optimal values

​	<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2011.49.11.png" alt="截屏2020-08-24 11.49.11" style="zoom: 67%;" />

1. **Policy evaluation**: calculate values for some **fixed policy** (not optimal values!) until convergence
2. **Policy improvement**: update policy using one-step look-ahead with resulting converged (but not optimal!) utilities as future values
3. Repeat steps until policy converges

### Policy Evaluation

**Value function** $V^\pi(s)$

- Expected long-term reward when being in state $\boldsymbol{s}$ and following policy $\pi(\boldsymbol{a}|\boldsymbol{s})$
  $$
  V^{\pi}(\boldsymbol{s})=\mathbb{E}\_{\pi, P}\left[\sum\_{t=0}^{\infty} \gamma^{t} r\_{t} \mid \boldsymbol{s}\_{0}=\boldsymbol{s}\right]
  $$

#### Example

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2011.55.49.png" alt="截屏2020-08-24 11.55.49" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2011.55.56.png" alt="截屏2020-08-24 11.55.56" style="zoom:67%;" />

#### Policy dependent Q-Function

**Q-Function** $Q^\pi(\boldsymbol{s}, \boldsymbol{a})$

- Long-term reward for taking action $\boldsymbol{a}$ in state $\boldsymbol{s}$ and subsequently following policy $\pi(\boldsymbol{a}|\boldsymbol{s})$
  $$
  Q^{\pi}(s, a)=\mathbb{E}\_{\pi, P}\left[\sum_{t=0}^{\infty} \gamma^{t} r\_{t} \mid s\_{0}=s, a\_{0}=a\right]
  $$

"How good" is it to take action $\boldsymbol{a}$ in state $\boldsymbol{s}$ and under policy $\pi(\boldsymbol{a}|\boldsymbol{s})$?

**V-Function and Q-Function can be computed from each other**

- V-Function is the expected Q-Function for a given state
  $$
  V^{\pi}(s)=\sum\_a \pi(a | s) Q^{\pi}(s, a)
  $$

- Q-Function is given by immediate reward plus discounted expected value of next state
  $$
  Q^{\pi}(\boldsymbol{s}, \boldsymbol{a})=r(\boldsymbol{s}, \boldsymbol{a})+\gamma \mathbb{E}\_{P}\left[V^{\pi}\left(\boldsymbol{s}^{\prime}\right) |\boldsymbol{s}, \boldsymbol{a}\right]
  $$

**By using these equations 👆, both functions can also be** **estimated recursively**

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2012.07.04.png" alt="截屏2020-08-24 12.07.04" style="zoom:80%;" />

#### Compute V's for a fixed policy $\pi$

- Use recursive Bellman equations for updates (like value iteration)

  ![截屏2020-08-24 12.13.37](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2012.13.37.png)

(Similarly, we can also **compute the Q-Function**)

### Policy Improvement / Policy Extraction

**Compute greedy action w.r.t value function of** $\pi$

- Using V-Function
  $$
  \pi\_{\text {new }}(\boldsymbol{s})=\underset{\boldsymbol{a}}{\arg \max }\left(r(\boldsymbol{s}, \boldsymbol{a})+\gamma \sum\_{\boldsymbol{s}^{\prime}} P\left(\boldsymbol{s}^{\prime} | \boldsymbol{s}, \boldsymbol{a}\right) V\_{k-1}^{\pi}\left(\boldsymbol{s}^{\prime}\right)\right)
  $$

- Using Q-Function
  $$
  \pi\_{\text {new }}(\boldsymbol{s})=\underset{\boldsymbol{a}}{\arg \max } Q^{\pi}(\boldsymbol{s}, \boldsymbol{a})
  $$

It can be shown that 
$$
Q^{\pi_{\text {new }}}(\boldsymbol{s}, \boldsymbol{a}) \geq Q^{\pi}(\boldsymbol{s}, \boldsymbol{a})
$$

### Policy Iteration Algorithm

![截屏2020-08-24 12.51.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2012.51.23.png)

### Policy Iteration vs. Value Iteration

- Both **value iteration** and **policy iteration** compute the same thing (all optimal values)
- **Both are** **dynamic programs** **for solving MDPs**
- **In policy iteration:**
  - We do several passes that update the value function with fixed policy
  - After the policy is evaluated, a new policy is chosen (slow like a value iteration pass)
  - The new policy will be better (or we’re done)
- **In value iteration:**
  - Every iteration updates both the values and (implicitly) the policy
  - We don’t track the policy, but taking the max over actions implicitly recomputes it
  - Extreme case of policy iteration where we stop the policy evaluation after one update

## 🔴 Problems

What we mentioned above does work. However, **Unfortunately, we can only do this in 2 cases**

- Discrete Systems
   ...but the world is not discrete! 😭

- Linear Systems, Quadratic Reward, Gaussian Noise (LQR) 

  ... but the world is not linear! 😭

**In all other cases,** **we have to use approximations!**

- **Representation of the V-function:** How to represent V in continuous state spaces?
- Need to solve
  - $\max \_{\boldsymbol{a}} Q^{\*}(\boldsymbol{s}, \boldsymbol{a})$: <span style="color:red">difficult in continuous action spaces</span>
  - $\mathbb{E}\_{\mathcal{P}}\left[V^{\*}\left(\boldsymbol{s}^{\prime}\right) | \boldsymbol{s}, \boldsymbol{a}\right]$: <span style="color:red">difficult for arbitrary functions V​ and models</span>

## Reference

- [RL slides from Berkeley](https://inst.eecs.berkeley.edu/~cs188/fa19/assets/slides/archive/SP18-CS188%20Lecture%208%20--%20MDPs%20I.pdf)