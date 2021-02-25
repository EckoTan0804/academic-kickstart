---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 11

# Basic metadata
title: "Advice on Reading Research Papers (by Prof. Andrew Ng)"
date: 2021-02-24
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Thesis", "How To", "Read Papers"]
categories: ["Thesis"]
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
    thesis:
        parent: read-papers
        weight: 2

---

Here we'll summarize two major recommendations given by Prof. Andrew Ng in his CS230 Deep Learning course:

- **Reading research papers**
- **Advice for navigating a career in machine learning**

## Reading research papers

### List of papers

- **Compile a list of papers**

  Try to create a list of research papers, [medium](https://medium.com/) posts and whatever text or learning resource you have

- **Skip around the list**
  - Read research papers in a parallel fashion; meaning try to tackle more than one paper at a time
  - Concretely, try to quickly skim and understand each of these paper and do not read it all, maybe you read 10–20% of each one and probably that will be enough to give you a high-level understanding of the paper in hand. 
  - After that, you may decide to eliminate some of these papers or just go over one or two them and read them fully.
- Amount of papers
  - **5–20 papers**
    - Probably enough basic knowledge of the specific domain
    - But maybe not enough to research or be at the cutting-edge.
  - **50–100 papers**
    - Probably have a very good understanding of the domain application

### How do you read ONE paper?

Do NOT start reading the paper from the first to the last word. Instead, **take multiple passes through the paper**

1. **Read the Title, the abstract and the figures**
   - By reading the title, abstract, the key network architecture figure, and maybe the experiments section, you will be able to get a general sense of the concepts in the paper.
   - In deep learning, there are a lot of research papers where the entire paper is summarized in one or two figures without the need to go hardly through the text.
2. **Read the introduction + conclusions + figures + skim the rest**
   - The **introduction**, the **conclusions** and the **abstract** are the places where the author(s) try to summarize their work carefully to clarify for the reviewer why their paper should be accepted for publication.
   - **Skim the related work section** (if possible), this section aims to highlight work done by others that somehow ties in with the author(s) work.
3. **Read the paper but skip the math**
4. **Read the whole thing but skip the parts that don’t make sense**

When reading a paper, try to answer the following questions:

- **What did the author(s) try to accomplish?**
- **What were the key elements of the approach?**
- **What can you use yourself?**
- **What other references do you want to follow?**

If you can answer these questions, hopefully, that will reflect that you have a good understanding of the paper.

{{% alert note %}} 

It turns out as you read more papers, with practice you get faster. Because a lot of authors use common formats when writing papers. 

{{% /alert %}}

### Deeper understanding

#### Math

Try to rederive it from scratch. Although, it takes some time but it’s a very good practice.

#### Code

1. Lightweight: Download open-source code (if you can find it) and run it.
2. Deeper: Reimplement from scratch. if you can do this, that’s a strong sign that you have really understood the algorithm in hand.

### To keep getting better

The most important thing to keep on learning and getting better is to **learn more steadily rather than having a focus-intensive activity**. 

>  It’s better to read two papers a week for the next year than cramming everything over a short period of time.

## Advice for navigating a career in machine learning

Just focus on doing important work and consider your job as a tactic and a chance to do useful work.

A very common pattern for successful machine learning engineers, **strong job candidates**, is to develop a **T-shaped knowledge base**

- have a *broad* understanding of many different topics in AI and 
- very *deep* understanding in at least one area.

![Image for post](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/1*p-4gmtKxINVGS8BOUQPwMg.jpeg)

**To build the horizontal piece**

A very efficient way to build foundational skills in these domains is through **courses and reading research papers**.

**To build the vertical piece**

You can build it by **doing related projects, open-source contributions, research and internships**.

### General advice

1. **Learn the most**

   tend to choose things to work on that allow you to learn the most.

2. **Do important work** 

   work on worthy projects that moves the world forward.

3. **Try to take machine learning to traditional industries** 

## Reference

- [Career advice/reading research papers](https://www.youtube.com/watch?v=733m6qBH-jI&list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb&index=9&t=0s) lecture in the CS230 Deep learning course by Stanford University

  {{< youtube 733m6qBH-jI>}}

- [Advice on building a machine learning career and reading research papers by Prof. Andrew Ng](https://blog.usejournal.com/advice-on-building-a-machine-learning-career-and-reading-research-papers-by-prof-andrew-ng-f90ac99a0182) - A summary for Prof. Andrew Ng's lecture