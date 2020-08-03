---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 310

# Basic metadata
title: "Generalization and Zeros"
date: 2020-08-03
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "Language models"]
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
        parent: LM-N-gram
        weight: 3

---

The n-gram model is dependent on the training corpus (like many statistical models).

Implication:

- The probabilities often encode specific facts about a given training corpus.
- n-grams do a better and better job of modeling the training corpus as we increase the value of $N$.

Notice when building n-gram models:

- use a training corpus that has a similar **genre** to whatever task we are trying to accomplish.

  - *To build a language model for translating legal documents, we need a training corpus of legal documents.*
  - *To build a language model for a question-answering system, we need a training corpus of questions.*

- Get training data in the appropriate dialect (especially when processing social media posts or spoken transcripts)

- Handle **sparsity**

  - When the corpus is limited, some perfectly acceptable English word sequences are bound to be missing from it.

    $\rightarrow$ <span style="color:red">We’ll have many cases of putative “zero probability n-grams” that should really have some non-zero probability! </span>

  - Example:

    - Consider the words that follow the bigram *denied the* in the WSJ Treebank3 corpus, together with their counts:

      ![截屏2020-06-03 12.03.38](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-03%2012.03.38-20200803105913667.png)

    - But suppose our test set has phrases like:

      ```
      denied the offer
      denied the loan
      ```

      Our model will incorrectly estimate that the $P(\text{offer}|\text{denied the})$ is 0! 🤪

  - **Zeros**: things that don’t ever occur in the training set but do occur in the test set

    - 🔴 Problems

      - We are **underestimating** the probability of all sorts of words that might occur, which will hurt the performance of any application we want to run on this data.

      - If the probability of any word in the test set is 0, the entire probability of the test set is 0.

        $\rightarrow$ Based on the definition of perplexity, we can’t compute perplexity at all, since we can’t divide by 0!

​				

### Unknow words

**Closed vocabulary** system: 

- All the words can occur
- the test set can only contain words from this lexicon, and there will be NO unknown words.
- Reasonable assumption in some domains
  - speech recognition (we have pronunciation dictionary in advance)
  - machine translation (we have phrase table in advance)
  - The language model can only use the words in that dictionary or phrase table.

**Unknown words**: words we simply have NEVER seen before.

- sometimes called **out of vocabulary (OOV)** words.
- **OOV rate**: percentage of OOV words that appear in the test set 

**Open vocabulary** system: 

- we model these potential unknown words in the test set by adding a pseudo-word called `<UNK>`.

Two common ways to to train the probabilities of the unknown word model `<UNK>`

- Turn the problem back into a closed vocabulary one by choosing a fixed vocabulary in advance

  1. **Choose a vocabulary** (word list) that is fixed in advance.

  2. **Convert** in the training set any word that is not in this set (any OOV word) to

     the unknown word token `<UNK>` in a text normalization step.

  3. **Estimate** the probabilities for `<UNK>` from its counts just like any other regular

     word in the training set.

- We don’t have a prior vocabulary in advance

  1. **Create** such a vocabulary implicitly

  2. **Replace** words in the training data by `<UNK>` based on their frequency.

     - we can replace by `<UNK>` all words that occur fewer than $n$ times in the training set, where $n$ is some small number, or
     - equivalently select a vocabulary size $V$ in advance (say 50,000) and choose the top  $V$ words by frequency and replace the rest by `<UNK>`

     In either case we then proceed to train the language model as before, treating `<UNK>` like a regular word.