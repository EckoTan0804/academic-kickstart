---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 220

# Basic metadata
title: "Words and Text Normalization"
date: 2020-08-02
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "Regular Expressions"]
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
        parent: re-tn-ed
        weight: 2

---

## TL;DR

- Two ways for counting words
  - Number of wordform types
    - Relationship between \#Types and \#Tokens: **Heap's law**
  - Number of lemmas
- Text Normalization
  1. Tokenizing (segmenting) words 
     - Bype-pair Encoding (BPE)
     - Wordpiece
  2. Normalizing word formats
     - Word normalisation
       - case folding
     - Lemmatization
     - Stemming 
       - Porter Stemmer
  3. Segmenting sentences



## Definition

- **Corpus** (pl. **corpora**): a computer-readable collection of text or speech. 

- **Lemma**: a set of lexical forms having the same stem, the same major part-of-speech, and the same word sense.
  - E.g.: `cats` and `cat` have the same lemma `cat`

- **Wordform**: full inflected or derived form of the word
  - E.g.: `cats` and `cat` have the same lemma `cat` but are different wordforms

How many words are there in English?

To answer this question we need to distinguish two ways of talking about words.

- One way: number of wordform types

  - **Type**:  number of distinct words in a corpus
    - if the set of words in the vocabulary is $V$ , the number of types is the vocabulary size $|V|$.
    
    - When we speak about the number of words in the language, we are generally referring to word types.

    - The larger the corpora we look at, the more word types we find
    
      
    
  - **Tokens**: total number $N$ of running words

    - E.g.: If we ignore punctuation, the following Brown sentence has 16 tokens and 14 types:

      

      ```tex
      They picnicked by the pool, then lay back on the grass and looked at the stars.
      ```
    
    
    
    - Relationship between the number of types $|V|$ and number of tokens $N$: **Herdan's Law** or **Heap's Law**
      $$
      |V|=k N^{\beta}
      $$
    
      - $k$: positive constant
      - $\beta \in (0, 1)$
        - depends on the corpus size and the genre
        - for the large corpora ranges from 0.67 to 0.75 

- Another way: number of lemmas
  
  - Dictionary **entries** or **boldface** forms are a very rough upper bound on the number of lemmas



## Text Normalization

three tasks are commonly applied as part of any normalization process:

1. [Tokenizing (segmenting) words](#word-tokenization) 
2. [Normalizing word formats](#word-nomalization-lemmatization-and-stemming)
3. [Segmenting sentences](#sentence-segmentation)

### Word Tokenization

**Tokenization**: tasks of segmenting running text into words.

For most NLP applications we’ll need to keep numbers and punctuation in our tokenization

- punctuation
  - as a separate token
    - commas `,`: useful piece of information for parsers
    - periods `.`: help indicate sentence boundaries
  - we also want to keep the punctuation that occurs word internally
    - E.g.: *m.p.h,*, *Ph.D.*, *AT&T*, *cap’n*
- Special characters and numbers need to be keep in 
  - prices *(\$45.55)* 
  - dates *(01/02/06)*
  - URLs *(http://www.stanford.edu)*
  - Twitter hashtags *(#nlproc)*
  - email address *(someone@cs.colorado.edu)*

A tokenizer can be used to 

- expand **clitic** contractions that are marked by apostrophes
  - `what're` -> `what are`

- **named entity detection**
  - tokenize multiword expressions like `New York` or `rock ’n’ roll` as a single token, which requires a multiword expression dictionary of some sort.

Commonly used tokenization standard: **Penn Treebank tokenization standard**

### Byte-Pair Encoding for Tokenization

💡 Instead of defining tokens as words (defined by spaces in orthographies that have spaces, or more complex algorithms), or as characters (as in Chinese), **we can use our data to automatically tell us what size tokens should be.**

**Morpheme**: smallest meaning-bearing unit of a language

- E.g.: the word `unlikeliest` has the morphemes `un-`, `likely`, and `-est`

One reason it’s helpful to have **subword** tokens is to deal with unknown words.

> Unknown words are particularly relevant for machine learning systems. Machine learning systems often learn some facts about words in one corpus (a training corpus) and then use these facts to make decisions about a separate test corpus and its words. Thus if our training corpus contains, say the words `low`, and `lowest`, but not `lower`, but then the word *lower* appears in our test corpus, our system will not know what to do with it. 🤪

🔧 Solution: use a kind of tokenization in which most tokens are words, but some tokens are frequent morphemes or other subwords like `-er`, so that an unseen word can be represented by combining the parts.

Simplest algorithm: **byte-pair encoding (BPE)**

- 💡 Intuition: iteratively merge frequent pairs of characters

- How it works?

  - Begins with the set of symbols equal to the set of characters.
    - Each word is represented as a sequence of characters plus a special end-of-word symbol `_`.
  - At each step of the algorithm, we count the number of symbol pairs, find the most frequent pair (‘A’, ‘B’), and replace it with the new merged symbol (‘AB’)
  - We continue to count and merge, creating new longer and longer character strings, until we’ve done $k$ merges ($k$ is a parameter of the algorithm)
  - The resulting symbol set will consist of the original set of characters plus $k$ new symbols.

- The algorithm is run *inside* words (we don’t merge across word boundaries). For this reason, the algorithm can take as input a dictionary of words together with counts.

- Example

  Consider the following tiny input dictionary with counts for each word,

  which would have the starting vocabulary of 11 letters

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-01%2011.50.28.png" alt="截屏2020-06-01 11.50.28" style="zoom:80%;" />

  - We first count all pairs of symbols: the most frequent is the pair (`r`,  `_`) because it occurs in *newer* (frequency of 6) and *wider* (frequency of 3) for a total of 9 occurrences. We then merge these symbols, treating `r_` as one symbol, and count again.

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-01%2011.52.05.png" alt="截屏2020-06-01 11.52.05" style="zoom:80%;" />

  - Now the most frequent pair is (`e`, `r_`) , which we merge; our system has learned that there should be a token for word-final `er`, represented as `er_` 

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-01%2011.53.38.png" alt="截屏2020-06-01 11.53.38" style="zoom:80%;" />

  - Next (`e`, `w`) (total count of 8) get merged to `ew`:

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-01%2011.54.53.png" alt="截屏2020-06-01 11.54.53" style="zoom:80%;" />

  - If we continue, the next merges are:

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-01%2011.55.22.png" alt="截屏2020-06-01 11.55.22" style="zoom:80%;" />

- Test
  - When we need to tokenize a test sentence, we just run the merges we have learned, greedily, in the order we learned them, on the test data. (Thus the frequencies in the test data don’t play a role, just the frequencies in the training data).
    - first we segment each test sentence word into characters.
    - Then we apply the first rule: replace every instance of `r`  `_`  in the test corpus with `r_` ; and then the second rule: replace every instance of `e` `r_` in the test corpus with `er_`, and so on.
    - By the end, if the test corpus contained the word `n e w e r _ `, it would be tokenized as a full word. But a new (unknown) word like `l o w e r _` would be merged into the two tokens `low` `er_` .

- In real algorithms BPE
  - run with many thousands of merges on a very large input dictionary
  - Result: most words will be represented as full symbols, and only the very rare words (and unknown words) will have to be represented by their parts.

#### Wordpiece and Greedy Tokenization

The **wordpiece** algorithm starts with some simple tokenization (such as by whitespace) into rough words, and then breaks those rough word tokens into subword tokens.

**Difference from BPE**:

- The special word-boundary token `_` appears at the **beginning** of words (rather than at the end)

- Rather than merging the pairs that are most *frequent*, wordpiece instead merges the pairs that minimizes the language model likelihood of the training data. 

  (the wordpiece model chooses the two tokens to combine that would give the training corpus the **highest** probability )

**How it works**?

- An input sentence or string is first split by some simple basic tokenizer (like whitespace) into a series of rough word tokens.

- Then instead of using a word boundary token, word-initial subwords are distinguished from those that do not start words by marking internal subwords with special symbols `##`

  - we might split `unaffable` into [`un`, `##aff"`, `##able`]

- Then each word token string is tokenized using a **greedy longest-match-first** algorithm.

  - Also called **maximum matching** or **MaxMatch**.

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-01%2012.28.54.png" alt="截屏2020-06-01 12.28.54" style="zoom:80%;" />

    - Given a vocabulary (a learned list of wordpiece tokens) and a string 
    - Starts by pointing at the beginning of a string
    - It chooses the longest token in the wordpiece vocabulary that matches the input at the current position, and moves the pointer past that word in the string. 
    - The algorithm is then applied again starting from the new pointer position.

**Example**: 

Given the token `intention` and the dictionary:

```
["in", "tent","intent","##tent", "##tention", "##tion", "#ion"]
```

The tokenizer would choose `intent` (because it is longer than `in`, and then `##ion` to complete the string, resulting in the tokenization `["intent" "##ion"]`.

### Word Normalization, Lemmatization and Stemming

- **Word normalization**: task of putting words/tokens in a standard format, choosing a single normal form for words with multiple forms
  - **Case folding**: Mapping everything to lower case
    - `Woodchuck` and `woodchuck` are represented identically
  - For many natural language processing situations we also want two morphologically different forms of a word to behave similarly.

- **Lemmatization**: task of determining that two words have the same root, despite their surface differences.
  - E.g.
    - `am`, `are`, and `is` have the shared lemma `be`
    - `dinner` and `dinners` both have the lemma *dinner*
    - The lemmatized form of a sentence like `He is reading detective stories` would thus be `He be read detective story`.
  - Method: complete **morphological parsing** of the word.
    - **Morphology**: study of the way words are built up from smaller meaning-bearing units called **morphemes**.
    - Two board classes of morphemes
      - **Stems**: the central morpheme of the word, supplying the main meaning
      - **Affixes**: adding "additional" meanings of various kinds
    - E.g.: 
      - the word `fox` consists of one morpheme (the morpheme `fox`) 
      - the word `cats` consists of two: the morpheme `cat` and the morpheme `-s`.

- **Stemming**: naive version of morphological analysis

  - Most widely used stemming algorithms: the **Porter Stemmer**

  - Based on series of rewrite rules run in series, as a **cascade**, in which the output of each pass is fed as input to the next pass

    Sampling of rules:

    <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-01%2012.49.59.png" alt="截屏2020-06-01 12.49.59" style="zoom:80%;" />

  - Simple stemmers can be useful in cases where we need to collapse across different variants of the same lemma

  - Nonetheless, they do tend to commit errors of both over- and under-generalizing 🤪

### Sentence Segmentation

The most useful cues for segmenting a text into sentences are **punctuation**

- Question marks and exclamation points are relatively unambiguous markers of sentence boundaries 👏
- Periods are more ambiguous 🤪
  - The period character “.” is ambiguous between a sentence boundary marker and a marker of abbreviations like `Mr.` or `Inc. (the final period of *Inc.* marked both an abbreviation and the sentence boundary marker 🤪)

Sentence tokenization methods work by first deciding (based on rules or machine learning) whether a period is part of the word or is a sentence-boundary marker.

- An abbreviation dictionary can help determine whether the period is part of a commonly used abbreviation



## Reference

- [Regular Expressions, Text Normalization, and Edit Distance](https://web.stanford.edu/~jurafsky/slp3/2.pdf)