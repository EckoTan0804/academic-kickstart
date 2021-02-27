---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 23

# Basic metadata
title: "Scientific Paper Structure"
date: 2021-02-26
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Thesis", "How To", "Write Papers"]
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
        parent: write-papers
        weight: 3

---

## The Sections of the Paper

Most journal-style scientific papers are subdivided into the following sections: **Title, Authors and Affiliation, Abstract, Introduction, Methods, Results, Discussion, Acknowledgments, and Literature Cited**, which parallel the experimental process.

| Section of Paper                                             | Experimental process         |
| ------------------------------------------------------------ | ---------------------------- |
| [Abstract]({{< relref "write-paper-abstract.md" >}})         | What did I do in a nutshell? |
| [Introduction]({{< relref "write-paper-introduction.md" >}}) | What is the problem?         |
| [Materials and Methods]({{< relref "write-paper-methods.md" >}}) | How did I solve the problem? |
| [Results]({{< relref "write-paper-results.md" >}})           | What did I find out?         |
| [Discussion]({{< relref "write-paper-discussion.md" >}})     | What does it mean?           |
| [Acknowledgements](#acknowledgements) (optional)             | Who helped me out?           |
| Literature Cited                                             | Whose work did I refer to?   |
| [Appendices](#appendices) (optional)                         | Extra Information            |

### Styles

**Main section headings**

- should be **capitalized**, **centered** at the beginning of the section
- Should be **double spaced** from the lines above and below
- Do NOT underline the section heading OR put a colon at the end. 

**Subheadings**

- Should be **capitalized** (first letter in each word), **left justified**, and either bold italics OR **underlined**.

<details>
<summary>Example of a subheading</summary>
    
***Effects of Light Intensity on the Rate of Electron Transport***
</details>

## Acknowledgements 

- If, in your experiment, you received any significant help in thinking up, designing, or carrying out the work, or received materials from someone who did you a favor by supplying them, you must acknowledge their assistance and the service or material provided.
- Authors always acknowledge **outside reviewers** of their drafts

- Place the Acknowledgments **between** the Discussion and the Literature Cited.

## Appendices

### Function

- Contains information that is
  - non-essential to understanding of the paper
  - but may present information that further clarifies a point without burdening the body of the presentation

- Optional part of the paper, and is only rarely found in published papers.

### Style

**Headings** 

- Each Appendix should be identified by a **Roman numeral in sequence**, e.g., Appendix I, Appendix II, etc. 
- Each appendix should contain different material.

### What could be put in appendix? (not an exhaustive list)

- raw data
- maps (foldout type especially)
- extra photographs
- explanation of formulas, either already known ones, or especially if you have "invented" some statistical or other mathematical procedures for data analysis.
- specialized computer programs for a particular procedure

## Reference

- [The Structure, Format, Content, and Style of a Journal-Style Scientific Paper](http://abacus.bates.edu/~ganderso/biology/resources/writing/HTWsections.html#abstract)