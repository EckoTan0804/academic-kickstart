---
# Title, summary, and position in the list
linktitle: "10-DM"
summary: ""
weight: 2100

# Basic metadata
title: "Dialog Management"
date: 2020-09-20
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "Lecture Notes"]
categories: ["NLP"]
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
    natural-language-processing:
        parent: lecture-notes
        weight: 11

---

![截屏2020-09-20 11.48.55](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2011.48.55.png)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2011.55.10.png" alt="截屏2020-09-20 11.55.10" style="zoom:80%;" />

## Dialog ModelingDialog Modeling

###  Dialog manager

- Manage flow of conversation

- Input: Semantic representation of the input

- Output: Semantic representation of the output

- Utilize additional knowledge 

  - User information

  - Dialog History 
  - Task-specific information

### 🔴 Challenges

- Consisting of many different components 
  - Each component has errors

  - More components --> less robust
- Should be modular

- Need to find unambiguous representation 
- Hard to train from data

## Dialog Types

### Goal-oriented Dialog

- Follows a fixed (set of) goals 
  - Ticket vending machines 
  - Restaurant reservation

  - Car SDS
- Aim: Reach goal as fast as possible
- **Main focus of SDS research**

### Social Dialog

- Social Dialog / Conversational Bots / Chit-Chat Setting 
- Most human

- Small talk conversation
- Aims:
  - Generate interesting, coherent, meaningful responses 
  - Carry-on as long as possible

  - Be a companion

## Dialog Systems

### Initiative

- **System Initiative **

  - Command & control

  - Example (U: User, S: System)

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2012.03.55.png" alt="截屏2020-09-20 12.03.55" style="zoom:70%;" />

- **Mixed Initiative**

  - Most nature

  - Example

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2012.05.30.png" alt="截屏2020-09-20 12.05.30" style="zoom:70%;" />

- **User Initiative**

  - User most powerful 

  - Error-prone

  - Example

    ![截屏2020-09-20 12.06.52](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2012.06.52.png)

### Confirmation

- **Explicit verification**

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2012.08.03.png" alt="截屏2020-09-20 12.08.03" style="zoom:67%;" />

- **Implicit verification**

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2012.08.25.png" alt="截屏2020-09-20 12.08.25" style="zoom:67%;" />

- **Alternative verification**

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2012.08.41.png" alt="截屏2020-09-20 12.08.41" style="zoom:67%;" />

### Development

- **Rule-based**
  - Create management by templates/rules

- **Statistical**
  - Train model to predict answer given input 
  - POMDP

- **End-to-End Neural Models**
  - No separation into NLU/DM/NLG

### Components

- **Dialog Model**: contains information about 
  - whether system, user or mixed initiative? 
  - whether explicit or implicit confirmation? 
  - what kind of speech acts needed?

- **User Model**: contains the system’s beliefs about 

  - what the user knows

  - the user’s expertise, experience and ability to understand the system’s utterances

- **Knowledge Base**: contains information about 
  - the world and the domain
- **Discourse Context**: contains information about 
  - the dialog history and the current discourse
- **Reference Resolver**
  - performs reference resolution and handles ellipsis

- **Plan Recognizer and Grounding Module**
  - interprets the user’s utterance given the current context 
  - reasons about the user’s goals and beliefs
- **Domain Reasoner/Planner**
  - generates plans to achieve the shared goals
- **Discourse Manager**
  - manages all information of dialog flow
- **Error Handling**
  - errors or misunderstandings detection and recovery



## Rule-based Systems

![截屏2020-09-20 13.42.02](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2013.42.02.png)

### Finite State-based

- 💡 **Idea: Iterate though states that define actions**

- Dialog flow:

  - specified as a set of dialog states (stages)

  - transitions denoting various alternative paths through the dialog graph 
  - **Nodes** = dialogue states (prompts)
  - Arcs = actions based on the recognized response

- Example

  ![截屏2020-09-20 12.57.29](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2012.57.29.png)

- 👍 Advantages
  - Simple to construct due to simple dialog control
  - The required vocabulary and grammar for each state can be specified in advance
    - Results in more constrained ASR and SLU
- 👎 Disadvantages
  - Restrict the user’s input to predetermined words/phrases 
  - Makes the correction of misrecognized items difficult
  - Inhibits the user’s opportunity to take the initiative and ask questions or introduce new topics

### Frame-based

- 💡 **Idea: Fill slots in a frame that defines the goal**

- Dialog flow:

  - is NOT predetermined, but depends on
    - the contents of the user’s input

    - the information that the system has to elicit

- Example

  - Eg1

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2013.12.50.png" alt="截屏2020-09-20 13.12.50" style="zoom:67%;" />

  - Eg2

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2013.13.34.png" alt="截屏2020-09-20 13.13.34" style="zoom:67%;" />

- Slot(/Form/Template) filling
  - One slot per piece of information

  - Takes a particular action based on the current state of affairs
- Questions and other prompts
  - List of possibilities
  - conditions that have to be true for that particular question or prompt
- 👍 Advantages
  - User can provide over-informative answers 
  - Allows more natural dialogues
- 👎 Disadvantages
  - Cannot handle complex dialogues

### Agent-based

- 💡 Idea:

  - Communication viewed as interaction between two agents 
  - Each capable of reasoning about its own actions and beliefs 
  - also about other’s actions and beliefs

  - Use of “contexts”

- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2013.20.28.png" alt="截屏2020-09-20 13.20.28" style="zoom:67%;" />

- Allow complex communication between the system, the user and the underlying application to solve some problem/task
- Many variants depends on particular aspects of intelligent behavior included
- Tends to be mixed-initiative
  - User can control the dialog, introduce new topics, or make contribution
- 👍 Advantages
  - Allow natural dialogue in complex domains
- 👎 Disadvantages
  - Such agents are usually very complex 
  - Hard to build :cry:

### Limitations of Rule-based DM

- Expensive to build Manual work
- Fragile to ASR errors

- No self-improvement over time



## Statistical DM

- Motivation

  - User intention can ONLY be imperfectly known 
    - Incompleteness – user may not specify full intention initially 
    - Noisiness – errors from ASR/SLU

  - Automatic learning of dialog strategies 
    - Rule based time consuming

- 👍 Advantages

  - Maintain a distribution over multiple hypotheses for the correct dialog state 
    - Not a single hypothesis for the dialog state

  - Choose actions through an automatic optimization process

  - Technology is not domain dependent
    - same technology can be applied to other domain by learning new domain data

### Markov Decision Process (MDP)

- A model for sequential decision making problems

  - Solved using **dynamic programming** and **reinforcement learning** 
  - MDP based SDM: dialog evolves as a **Markov process**

- Specified by a tuple $(S, A, T, R)$

  - $S$: a set of possible world states $s \in S$

  - $A$: a set of possible actions $a\in A$

  - $R$: a local real-valued reward function  
    $$
    R: S \times A \mapsto \mathcal{R}
    $$

  - $T$: a transition mode 
    $$
    T(s\_{t-1}, a\_{t-1}, s\_t) = P(s\_t | s\_{t-1}, a\_{t-1})
    $$

- 🎯 Goal of MDP based SDM: Maximize its expected cumulative (discounted) reward
  $$
  E\left(\sum\_{t=0}^{\infty} \gamma^{t} R\left(s\_{t}, a\_{t}\right)\right)
  $$

- Requires complete knowledge of $S$ !!!

### Reinforcement Learning

- **“Learning through trial-and-error” (reward/penalty)**
- 🔴 Problem
  - No direct feedback

  - Only feedback at the end of dialog

- 🎯 Goal: Learn evaluation function from feedback
- 💡 Idea
  - Initial all operations have equal probability

  - If dialog was successful --> all operations are positive 
  - If dialog was negative --> operations negative

#### How RL works?

- There is an **agent** with the capacity to **act**

- Each **action** influences the agent’s future **state**

- Success is measured by a scalar **reward** signal

- In a nutshell:

  - **Select actions to maximize future reward**

  - Ideally, a single agent could learn to solve any task :muscle:

#### Sequential Decision Making

- 🎯 **Goal: select actions to maximize total future reward**
- Actions may have long term consequences
- Reward may be delayed
- It may be better to sacrifice immediate reward to gain more long-term reward 🤔

#### Agent and Environment

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2015.50.33.png" alt="截屏2020-09-20 15.50.33" style="zoom:50%;" />

At each step $t$

- Agent:
  - Receives state $s\_t$ 
  - Receives scalar reward $r\_t$ 
  - Executes action $a\_t$
- The environment:
  - Receives action $a\_t$ 
  - Emits state $s\_t$
  - Emits scalar reward $r\_t$
-  The evolution of this process is called a **Markov Decision Process (MDP)**

#### Supervised Learning Vs. Reinforcement Learning

**Supervised Learning**:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2016.04.21.png" alt="截屏2020-09-20 16.04.21" style="zoom:100%;" />

- Label is given: we can compute gradients given label and update our parameters

**Reinforcement Learning**

![截屏2020-09-20 16.05.11](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-09-20%2016.05.11.png)

- NO label given: instead we have feedback from the environment
- Not an absolute label / error. We can compute gradients, but do not yet know if our action choice is good. 🤪

{{% alert note %}} 

More see: [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)

{{% /alert %}}

#### Policy and Value Functions

- **Policy $\pi$** : a probability distribution of actions given a state
  $$
  a = \pi(s)
  $$

- **Value function $Q^\pi(s, a)$** : the expected total reward from state $s$ and action $a$ under policy $\pi$
  $$
  Q^{\pi}(s, a)=\mathbb{E}\left[r\_{t+1}+\gamma r\_{t+2}+\gamma^{2} r\_{t+3}+\cdots \mid s, a\right]
  $$

  - “How good is action $a$ in state $s$?”
    - Same reward for two actions, but different consequences down the road
    - Want to update our value function accordingly

#### Appoaches to RL

- **Policy-based RL**

  - Search directly for the **optimal policy $\pi^\*$**

    (policy achieving maximum future reward)

- **Value-based RL**

  - Estimate the **optimal value function $Q^{∗}(s,a)$**
    (maximum value achievable under any policy)
  - **Q-Learning**: Learn Q-Function that approximates $Q^{∗}(s,a)$
    - Maximum reward when taking action $a$ in $s$
    - Policy: Select action with maximal $Q$ value
    - Algorithm:
      - Initialized $Q$ randomly
      - $Q(s, a) \leftarrow(1-\alpha) Q(s, a)+\alpha\left(r\_{t}+\gamma \cdot \underset{a}{\max} Q\left(s\_{t+1}, a\right)\right)$