---
# Title, summary, and position in the list
linktitle: "RL Basics"
summary: ""
weight: 810

# Basic metadata
title: "Reinforcement Learning Basics"
date: 2020-08-23
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
        weight: 1

---

## Overview

**Reinforcement Learning: <span style="color:orange">Learn</span> to choose a <span style="color:green">good</span> <span style="color:blue">sequence of actions</span>**

- <span style="color:blue">sequence of actions: Repeated interaction with the world</span> 

- <span style="color:green">good: There is an optimality criterion</span> 

- <span style="color:orange">Learn: No (or limited) knowledge how the world works</span>

## Definition

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2000.12.41.png" alt="截屏2020-08-24 00.12.41" style="zoom: 33%;" />

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
Learning: Adaption of policy $\pi(\boldsymbol{a}|\boldsymbol{s})$

### Example: Grid World

![截屏2020-08-24 00.18.29](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2000.18.29.png)

![截屏2020-08-24 00.19.31](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2000.19.31.png)

### Reward

- Reward $r\_t$: a scalar feedback signal
- Evaluates how “good” an agent acted at time step $t$

### Goal of the Agent

🎯 **Find a policy which maximizes the expected cumulative future reward (aka. return)**
$$
J\_{\pi}=\mathbb{E}\_{\pi}[\underbrace{\sum\_{t=0}^{\infty} \gamma^{t} r\_{t}}\_{\text {return } R}]
$$

- $0 \leq \gamma < 1$: Discounting factor
- Trade-off between optimizing the long-term ( $\gamma \to 1$) und short-term ( $\gamma \to 0$) Reward

### Sequential Interaction

- Actions have long-term effects
- Rewards can be delayed
- **Less immediate reward may yield higher long-term reward**
- Examples:
  - Investment: long duration until profit is made
  - Fuelling a helicopter: avoid crash in several hours
  - Blocking an opponent: might increase chances of winning in the future

### Components of a RL Agent

**A RL agent consists of *at least* one of the following components:**

- **Policy:** Behaviour of the Agent
- **Value-Function:** Evaluation-function for states and actions
- **Model:** Agent’s representation of the environment

#### Policy

![截屏2020-08-24 00.25.06](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2000.25.06.png)

- Represents the behaviour of the agent

- Selects action given state

- **Deterministic:**
  $$
  \boldsymbol{a} = \pi(\boldsymbol{s})
  $$

- **Stochastic**:
  $$
  \pi(\boldsymbol{a}|\boldsymbol{s})
  $$

- 

#### Value Functions

![截屏2020-08-24 00.31.18](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2000.31.18.png)

- **V-Function:** Expected future reward if agent is in state $s$ and follows policy $\pi$
  $$
  V^{\pi}(\boldsymbol{s})=\mathbb{E}\_{\pi}\left[\sum\_{t=0}^{\infty} \gamma^{t} r\_{t} \mid \boldsymbol{s}\_{0}=\boldsymbol{s}\right]
  $$
  (**Quality metric for states**)

- **Q-Function:** Expected future reward if 

  - agent in state $\boldsymbol{s}$,

  - chooses action $\boldsymbol{a}$ and 
  - follows policy $\pi$ afterwards

  $$
  Q^{\pi}(\boldsymbol{s},\boldsymbol{a})=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} r_{t} \mid \boldsymbol{s}_{0}=\boldsymbol{s}, \boldsymbol{a}_{0}=\boldsymbol{a}\right]
  $$

  (**Quality metric for state-action pairs**)

#### Model

![截屏2020-08-24 00.32.03](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2000.32.03.png)

- **Internal** representation of the agent of the environment

- **Prediction**
  - Transition-Model:
    $$
    \hat{p}\left(\boldsymbol{s}^{\prime} | \boldsymbol{s}, \boldsymbol{a}\right) \approx p\left(\boldsymbol{s}^{\prime} | \boldsymbol{s}, \boldsymbol{a}\right)
    $$

  - Reward-Model:
    $$
    \hat{r}(\boldsymbol{s}, \boldsymbol{a}) \approx r(\boldsymbol{s}, \boldsymbol{a})
    $$

- Learned from data
- Error-prone 🤪

## Taxonomy of RL Algorithms

![截屏2020-08-24 00.35.23](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2000.35.23.png)

**Different tradeoffs:**

- Sample efficiency
- Stability & ease of use

**Different assumptions:**

- Stochastic or deterministic?
- Continuous or discrete?
- Episodic or infinite horizon?

**Different things are easy or hard in different settings**

- Easier to represent the policy?
- Easier to represent the model?

### Comparison: Sample efficiency

**Sample efficiency = how many samples do we need to get a good policy?**

- Most important question: is the algorithm off policy?

> - **Off policy:** able to improve the policy WITHOUT generating new samples from that policy (i.e. reuse experience from old policies)
> - **On policy:** each time the policy is changed, even a little bit, we need to generate new samples

![截屏2020-08-24 00.39.05](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-24%2000.39.05.png)

 **We want to use a less sample efficient algorithm** because of

- Stability and easy of use
  - Does it converge?
  - And if it converges, to what?
  - And does it converge every time?
- Scalability
- Quality of the final policy
- Wall-clock time (not the same as sample complexity!)