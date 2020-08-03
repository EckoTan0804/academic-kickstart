---
# Title, summary, and position in the list
linktitle: "POS-Tagging"
summary: ""
weight: 810

# Basic metadata
title: "Part-of-Speech Tagging"
date: 2020-08-03
draft: false
type: docs # page type
authors: ["admin"]
tags: ["NLP", "POS tagging"]
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
        parent: POS-tagging
        weight: 1

---

Parts of speech (a.k.a **POS**, **word classes**, or **syntactic categories**) are useful because they reveal a lot about a word and its neighbors

- Knowing whether a word is a noun or a verb tells us about likely neighboring words

  

## Style Convention

- **Bold**: 

  - new terms/concept/definition

  - important points

    

- *Italic*: examples

  

## (Mostly) English Word Classes

Parts of Speech 

- **closed** class

  - close $\equiv$ relatively fixed membership

  - Generally **function words**, which
    - tend to be very short
    - occur frequently
    - often have structuring uses in grammar

  - Important closed class in English:

    - **Prepositions**
      - *on, under, over, near, by, at, from, to, with*
      - Occur before noun phrases
      - Often indicate spatial or temporal relations
        - literal (*on it*, *before then*, *by the house*)
        - metaphorical (*on time*, *with gusto*, *beside herself*)
      - Also indicate other relations
        - *Hamlet was written <u>by</u> Shakespeare*
    - **Particle**
      - *up, down, on, off, in, out, at, by*
      - Resembles a preposition or an adverb 
      - Used in combination with a verb
      - Particles often have extended meanings that aren’t quite the same as the prepositions they resemble
    - **Determiner**
      - Occurs with nouns, often marking the beginning of a noun phrase
        - **article**
          - indefinite: *a, an*
          - definite: *the*
        - Other determiner: *this, that* (*this chapter, that page*)
    - **Conjunctions**: join two phrases, clauses, or sentences
      - **Coordinating conjunctions**: join two elements of equal status
        - *and, or, but*
      - **Subordinating conjunctions**: when one of the elements has some embedded status
        - *I thought that you might like some milk*
          - main clause: *I thought*
          - subordinate clause: *you might like some milk*
          - Subordinating conjunctions like *that* which link a verb to its argument in this way are also called **complementizers**.
    - **Pronouns**: often act as a kind of shorthand for referring to some noun phrase or entity or event
      - **Personal pronouns**: refer to persons or entities 
        - *you*, *she*, *I*, *it*, *me*, etc 
      - **Possessive pronouns**: forms of personal pronouns that indicate either actual possession or more often just an abstract relation between the person and some object 
        - *my, your, his, her, its, one’s, our, their*
      - **Wh-pronouns**: are used in certain question forms
        - *what, who, whom, whoever*
        - may also act as complementizers
          - *Frida, who married Diego. . .*

    - **auxiliary verbs**
      - mark semantic features of a main verb
        - whether an action takes place in the present, past, or future (tense)
        - whether it is completed (aspect)
        - whether it is negated (polarity)
        - whether an action is necessary, possible, suggested, or desired (mood)
      - **Copula verb**: 
        - *be*
          - connects subjects with certain kinds of predicate nominals and adjectives
            - *He <u>is</u> a duck*
          - is used as part of the passive (*We <u>were</u> robbed*) or progressive (*We <u>are</u> leaving*) constructions
        - *have* 
          - mark the perfect tenses
            - *I <u>have</u> gone*
            - *I <u>had</u> gone*
      - **Modal verbs**: mark the mood associated with the event depicted by the main verb
        - *can*: indicates ability or possibility
        - *may*: indicates permission or possibility
        - *must*: indicates necessity
        - There is also a modal use of *have* (e.g., *I <u>have</u> to go*).

- **open** class

  - open $\equiv$ words are continually being created or borrowed 
  - Four major open classes:
    - **Nouns**: include concrete terms (*ship* and *chair*), abstractions (*bandwidth* and *relationship*), and verb-like terms (*pacing* as in *His pacing to and fro became quite annoying*)
      - **Proper nouns**: names of specific persons or entities
        - E.g.: *Regina, Colorado, IBM*
        - Generally NOT preceeded by articles
          - *Regina is upstairs*
        - Usually capitalized
      - **Common nouns**
        - **Count nouns**
          - Allow grammatical enumeration, occurring in both the singular and plural 
            - *goat/goats, relationship/relationships*
          - Can be counted
            - *one goat, two goats*
          - Singular count nouns can NOT appear without articles
            - ~~*Goat is white*~~ ❌
        - **Mass nouns**
          - something is conceptualized as a homogeneous group
          - Can NOT be counted
            - *snow, salt*, and *communism*
            - *~~two snows~~* ❌ 
          - Can appear without articles
            - *Snow is white*
    - **Verbs**
      - Refer to actions and processes, including main verbs (*draw, provide, go*)
      - Have infections
        - non-third-person-sg (*eat*)
        - third-person-sg (*eats*)
        - progressive (*eating*)
        - past participle (*eaten*)
    - **Adjectives**
      - Includes many terms for properties or qualities
    - **Adverbs**: can be viewes as modifying something (often verbs)
      - Type:
        - **Directional/locative adverbs**: specify the direction or location of some action
          - *home*, *here*, *downhill*
        - **Degree adverbs**: specify the extent of some action. process, or property
          - *extremely*, *very*, *somewhat*
        - **Manner adverbs**: describe the manner of some action or process
          - *slowly*, *slinkily*, *delicately*
        - **Temporal adverbs**: describe the time that some ac- tion or event took place
          - *yesterday*, *Monday*
      - Some adverbs (e.g., temporal adverbs like *Monday*) are tagged in some tagging schemes as nouns.

- Many words of more or less unique function

  - **interjections** (*oh, hey, alas, uh, um*), 

  - **negatives** (*no, not*), 

  - **politeness markers** (*please, thank you*), 

  - **greetings** (*hello, goodbye*), 

  - existential **there** (*there are two on the table*)

    These classes may be distinguished or lumped together as interjections or adverbs depending on the purpose of the labeling.

### Summary

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-23%2015.40.32.png" alt="截屏2020-05-23 15.40.32" style="zoom: 40%;" />



## The Penn Treebank Part-of-Speech Tagset

- 45-tag Penn Treebank tagset (Marcus et al., 1993)
- Parts of speech are generally represented by placing the tag after each word, delimited by a slash

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-23%2015.42.40.png" alt="截屏2020-05-23 15.42.40" style="zoom:80%;" />

Example: 

- The/DT grand/JJ jury/NN commented/VBD on/IN a/DT number/NN of/IN other/JJ topics/NNS ./.

- There/EX are/VBP 70/CD children/NNS there/RB

- Preliminary/JJ findings/NNS were/VBD reported/VBN in/IN today/NN

  ’s/POS New/NNP England/NNP Journal/NNP of/IN Medicine/NNP ./.

### Tagged corpora

- **Brown** corpus
- **WSJ** corpus
- **Switchboard** corpus



## Part-of-Speech Tagging

**Part-of-speech tagging**: process of assigning a part-of-speech marker to each word tokens in an input text

- Input to a tagging algorithm: a sequence of (tokenized) word in an input text. words and a tagset, 

- output: a sequence of tags, one per token.

**Tagging**: disambiguation task

- Words are ambiguous: have more than one possible part-of-speech 😧

- 🎯 Goal: find the correct tag for the situation

- E.g.

  - *book* can be a verb (*<u>book</u> that flight*) or a noun (*hand me that <u>book</u>*)

  - *That* can be a determiner (Does that flight serve dinner) or a complementizer (*I*

    thought that your flight was earlier).

**The goal of POS-tagging is to resolve these ambiguities, choosing the proper tag for the context.** 💪

Most ambiguous frequent words are *that*, *back*, *down*, *put* and *set*. 

E.g., 6 POS for the word *back*

- earnings growth took a **back/JJ** seat

- a small building in the **back/NN**

- a clear majority of senators **back/VBP** the bill 
- Dave began to **back/VB** toward the door 
- enable the country to buy **back/RP** about debt 
- I was twenty-one **back/RB** then

Nonetheless, many words are easy to disambiguate, because their different tags aren’t equally likely. This idea suggests a simplistic baseline algorithm for part-of-speech tagging: **given an ambiguous word, choose the tag which is most frequent in the training corpus.** 💡

> **Most Frequent Class Baseline**: Always compare a classifier against a baseline at least as good as the most frequent class baseline (assigning each token to the class it occurred in most often in the training set).

**How good is this baseline?** 

- Standard way to measure the performance of part- of-speech taggers: **accuracy**
  - the percentage of tags correctly labeled (matching human labels on a test set)
- train on the WSJ training corpus and test on sections 22-24 of the same corpus 
  - the most-frequent-tag baseline achieves an accuracy of 92.34%.
  - the state of the art in part-of-speech tagging on this dataset is around 97% tag accuracy