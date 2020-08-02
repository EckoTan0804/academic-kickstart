# N-gram Language Models

**Language models (LMs)**: Model that assign probabilities to sequence of words

**N-gram**: a sequence of N words

​	E.g.: *Please turn your homework ...*

- **bigram (2-gram)**: two-word sequence of word
  - *“please turn”*, *“turn your”*, or *”your homework”*
- **trigram (3-gram)**: three-word sequence of word
  - *“please turn your”*, or *“turn your homework”*



## N-Grams

$P(w|h)$: probability of a word $w$ given some history $h$.

Our task is to compute $P(w|h)$.

Consider a simple example: 

Suppose the history $h$ is “*its water is so transparent that*” and we want to know the probability that the next word is *the*: $P(\text {the} | \text {its water is so transparent that})$

### **Naive way**

Use **relative frequency counts** (“Out of the times we saw the history *h*, how many times was it followed by the word *w*”?)

- Take a very large corpus, count the number of times we see *its water is so transparent that*, and count the number of times this is followed by *the*. 

$$
P(\text {the} | \text {its water is so transparent that})=
\frac{C(\text {its water is so transparent that the})}{C(\text {its water is so transparent that})}
$$

- With a large enough corpus, such as the web, we can compute these counts and estimate the probability
  - Works fine in many cases
- 🔴 Problems
  - Even the web isn’t big enough to give us good estimates in most cases. 
    - This is because language is creative; new sentences are created all the time, and we won’t always be able to count entire sentences.
  - Similarly, if we wanted to know the joint probability of an entire sequence of words like *its water is so transparent*, we could do it by asking “out of all possible sequences of five words, how many of them are *its water is so transparent*?”
    - We have to get the count of *its water is so transparent* and divide by the sum of the counts of all possible five word sequences. That seems rather a lot to estimate!

### Cleverer way

Notation:

- $P(X_i=\text{''the''})$: probability of a particular random variable $X_i$ taking on the value “the”
  - Simplification: $P(the)$

- $w_1\dots w_n$ or $w_1^n$: a sequence of $n$ words
  - $w_1^{n-1}$: the string $w_1, w_2, \dots w_{n-1}$
- $P(w_1, w_2, \dots, w_n)$: joint probability of each word in a sequence having a particular value $P(X_1=w_1, X_2=w_2, \dots, X_n=w_n)$

Compute $P(w_1, w_2, \dots, w_n)$: Use the **chain rule of probability**
$$
\begin{aligned}
P\left(X_{1} \ldots X_{n}\right) &=P\left(X_{1}\right) P\left(X_{2} | X_{1}\right) P\left(X_{3} | X_{1}^{2}\right) \ldots P\left(X_{n} | X_{1}^{n-1}\right) \\
&=\prod_{k=1}^{n} P\left(X_{k} | X_{1}^{k-1}\right)
\end{aligned}
$$
Apply to words:
$$
\begin{aligned}
P\left(w_{1}^{n}\right) &=P\left(w_{1}\right) P\left(w_{2} | w_{1}\right) P\left(w_{3} | w_{1}^{2}\right) \ldots P\left(w_{n} | w_{1}^{n-1}\right) \\
&=\prod_{k=1}^{n} P\left(w_{k} | w_{1}^{k-1}\right)
\end{aligned}
$$
🔴 Problem: We don’t know any way to compute the exact probability of a word given a long sequence of preceding words $P(w_n|w_1^{n-1})$

- we can’t just estimate by counting the number of times every word occurs following every long string, because language is creative and any particular context might have never occurred before! 🤪

🔧 Solution: **n-gram model**

💡 Idea of n-gram model: instead of computing the probability of a word given its entire history, we can **approximate the history by just the last few words**.

E.g.: the **bigram** model, approximates the probability of a word given all the previous words $P(w_n|w_1^{n-1})$ by using only the conditional probability of the PRECEDING word $P(w_n|w_{n-1})$:
$$
P\left(w_{n} | w_{1}^{n-1}\right) \approx P\left(w_{n} | w_{n-1}\right)
$$

- E.g.: $P(\text { the } | \text { Walden Pond's water is so transparent that }) \approx P(\text{the}|\text{that})$

👆 The assumption that the probability of a word depends only on the previous word is called a **Markov assumption**. Markov models are the class of probabilistic models that assume we can predict the probability of some future unit *without* looking too far into the past. 

Generalize the bigram (which looks one word into the past) to the trigram (which looks two words into the past) and thus to the n-gram (which looks $n − 1$ words into the past):
$$
P\left(w_{n} | w_{1}^{n-1}\right) \approx P\left(w_{n} | w_{n-N+1}^{n-1}\right)
$$

### Estimate n-gram probabilities

Intuitive way: **Maximum Likelihood Estimation (MLE)**

- Get counts from a corpus
- Normalize the counts so that they lie between 0 and 1

#### Bigram

Let's start from bigram. To compute a particular bigram probability of a word $y$ given a previous word $x$, we'll compute the count of the bigram $C(xy)$ and normalize by the sum of all the bigrams that share the same first word $x$
$$
P\left(w_{n} | w_{n-1}\right)=\frac{C\left(w_{n-1} w_{n}\right)}{\sum_{w} C\left(w_{n-1} w\right)}
$$
We can simplify this equation, since the sum of all bigram counts that start with a given word $w_{n-1}$ must be equal to the unigram count for that word $w_{n-1}$
$$
P\left(w_{n} | w_{n-1}\right)=\frac{C\left(w_{n-1} w_{n}\right)}{C\left(w_{n-1}\right)}
$$
**Example**: 

Given a mini-corpus of three sentences

```
<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
```

- We need to augment each sentence with a special symbol `<s>` at the beginning of the sentence, to give us the bigram context of the first word.

The calculations for some of the bigram probabilities from this corpus:
$$
\begin{array}{lll}
P(\mathrm{I} |<\mathrm{s}>)=\frac{2}{3}=0.67 & P(\mathrm{Sam} |<\mathrm{s}>)=\frac{1}{3}=0.33 & P(\mathrm{am} | \mathrm{I})=\frac{2}{3}=0.67 \\
P(</ \mathrm{s}>| \mathrm{Sam})=\frac{1}{2}=0.5 & P(\mathrm{Sam} | \mathrm{am})=\frac{1}{2}=0.5 & P(\mathrm{do} | \mathrm{I})=\frac{1}{3}=0.33
\end{array}
$$

#### N-gram

For the general case of MLE n-gram parameter estimation:
$$
P\left(w_{n} | w_{n-N+1}^{n-1}\right)=\frac{C\left(w_{n-N+1}^{n-1} w_{n}\right)}{C\left(w_{n-N+1}^{n-1}\right)}
$$
It estimates the n-gram probability by dividing the observed frequency of a particular sequence by the observed frequency of a prefix. This ratio is called a **relative frequency**.

**Example**: 

Use data from the now-defunct Berkeley Restaurant Project.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-02%2016.56.21.png" alt="截屏2020-06-02 16.56.21" style="zoom:80%;" />

👆 This figure shows the bigram counts from a piece of a bigram grammar.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-02%2016.58.56.png" alt="截屏2020-06-02 16.58.56" style="zoom:80%;" />

👆 This figure shows the bigram probabilities after normalization (dividing each cell in figure above (Figure 3.1) by the appropriate unigram for its row, taken from the following set of unigram probabilities)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-02%2016.59.42.png" alt="截屏2020-06-02 16.59.42" style="zoom:80%;" />

Other useful probabilities:
$$
\begin{array}{ll}
P(\mathrm{i} |<\mathrm{s}>)=0.25 & P(\text { english } | \text { want })=0.0011 \\
P(\text { food } | \text { english })=0.5 & P(</ \mathrm{s}>| \text { food })=0.68
\end{array}
$$
Now we can compute the probability of sentences like *I want English food* by simply multiplying the appropriate bigram probabilities together:
$$
\begin{aligned}
&P(\langle s\rangle\text { i want english food}\langle / s\rangle) \\
=\quad  & P(\mathrm{i} |<\mathrm{s}>) \cdot P(\text { want } | \mathrm{i}) \cdot P(\text { english } | \text { want }) \cdot P(\text { food } | \text { english }) \cdot P(</ \mathrm{s}>| \text { food }) \\
=\quad & .25 \times .33 \times .0011 \times 0.5 \times 0.68 \\
=\quad & .000031
\end{aligned}
$$

#### Pratical issues

- In practice it’s more common to use **trigram** models, which condition on the previous two words rather than the previous word, or **4-gram** or even **5-gram** models, when there is sufficient training data.
  - Note that for these larger n- grams, we’ll need to assume extra context for the contexts to the left and right of the sentence end. For example, to compute trigram probabilities at the very beginning of the sentence, we can use two pseudo-words for the first trigram (i.e., $P(I|<s><s>)$.

- We always represent and compute language model probabilities in log format as **log probabilities**.
  - Multiplying enough n-grams together would easily result in **numerical underflow** 🤪 (Since probability $\in (0, 1)$)
  - Adding in log space is equivalent to multiplying in linear space, so we combine log probabilities by adding them. 



## Evaluating Language Models

**Extrinsic evaluation**

- Best way to evaluate the performance of a language model

- Embed LM in an application and measure how much the application improves
- For speech recognition, we can compare the performance of two language models by running the speech recognizer twice, once with each language model, and seeing which gives the more accurate transcription.
- 🔴 Problem: running big NLP systems end-to-end is often very expensive

**Intrinsic evaluation**: 

- measures the quality of a model independent of any application.

- Can be used to quickly evaluate potential improvements in a language model
- We need
  - **Training set (Training corpus)**
  - **Development test set (devset)**
    - Also called **Validation set** (see [wiki](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets))
    - Particular test set
      - Implicitly tune to its characteristics
  - **Test set (Test corpus)**
    - NOT to let the test sentences into the training set!
    - Truely UNSEEN!
  - In practice: divide data into 80% training, 10% development, and 10% test.
- How it works?
  - Given a corpus of text and we want to compare two different n-gram models
    1. we divide the data into training and test sets, 
    2. train the parameters of both models on the training set, and 
    3. then compare how well the two trained models fit the test set.
       - "Fit the test set" means: whichever model assigns a **higher probability** to the test set—meaning it more accurately predicts the test set—is a **better** model.

### Perplexity

Instead of raw probability as our metric for evaluating language models, in practice we use **perplexity**.

The **perplexity** (sometimes called ***PP*** for short) of a language model on a test set is the inverse probability of the test set, normalized by the number of words.

For a test set $W=w_{1} w_{2} \ldots w_{N}$:
$$
\begin{array}{ll}
\operatorname{PP}(W) &=P\left(w_{1} w_{2} \ldots w_{N}\right)^{-\frac{1}{N}} \\
&=\sqrt[N]{\frac{1}{P\left(w_{1} w_{2} \ldots w_{N}\right)}} \\
&\overset{\text{chain rule}}{=} \sqrt[N]{\displaystyle\prod_{i=1}^{N} \frac{1}{P\left(w_{i} | w_{1} \ldots w_{i-1}\right)}}
\end{array}
$$
Thus, perplexity of *W* with a bigram language model is
$$
\operatorname{PP}(W)=\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P\left(w_{i} | w_{i-1}\right)}}
$$
The higher the conditional probabil- ity of the word sequence, the lower the perplexity. Thus, minimizing perplexity is equivalent to maximizing the test set probability according to the language model.

What we generally use for word sequence in perplexity computation is the ENTIRE sequence of words in test test. Since this sequence will cross many sentence boundaries, we need to include 

- the begin- and end-sentence markers `<s>` and `</s>` in the probability computation. 
- the end-of-sentence marker `</s>` (but not the beginning-of-sentence marker `<s>`) in the total count of word tokens *N*.

#### Another aspect 

We can also think about perpleixty as the **weighted average branching factor** of a language.

- branching factor of a language: the number of possible next words that can follow any word.

Example

Consider the task of recognizing the digits in English (zero, one, two,..., nine), given that (both in some training set and in some test set) each of the 10 digits occurs with equal probability $P=\frac{1}{10}$. The perplexity of this mini-language is in fact 10. 
$$
\begin{aligned}
\operatorname{PP}(W) &=P\left(w_{1} w_{2} \ldots w_{N}\right)^{-\frac{1}{N}} \\
&=\left(\frac{1}{10}^{N}\right)^{-\frac{1}{N}} \\
&=\frac{1}{10}^{-1} \\
&=10
\end{aligned}
$$
Now suppose that the number zero is really frequent and occurs far more often than other numbers.

- 0 occur 91 times in the training set, and 
- each of the other digits occurred 1 time each.

Now we see the following test set: `0 0 0 0 0 3 0 0 0 0`. We should expect the perplexity of this test set to be lower since most of the time the next number will be zero, which is very predictable (i.e. has a high probability).  Thus, although the branching factor is still 10, the perplexity or *weighted* branching factor is smaller. 



## Generalization and Zeros

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

      ![截屏2020-06-03 12.03.38](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-03 12.03.38.png)

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

  1. Create such a vocabulary implicitly

  2. Replace words in the training data by `<UNK>` based on their frequency.

     - we can replace by `<UNK>` all words that occur fewer than $n$ times in the training set, where $n$ is some small number, or
     - equivalently select a vocabulary size $V$ in advance (say 50,000) and choose the top  $V$ words by frequency and replace the rest by `<UNK>`

     In either case we then proceed to train the language model as before, treating `<UNK>` like a regular word.



## Smoothing

To keep a language model from assigning zero probability to these unseen events, we’ll have to shave off a bit of probability mass from some more frequent events and give it to the events we’ve never seen.

### Laplace smoothing (Add-1 smoothing)

💡 Idea: add one to all the bigram counts, before we normalize them into probabilities. 

- does not perform well enough to be used in modern n-gram models 🤪, but
- usefully introduces many of the concepts
- gives a useful baseline
- a practical smoothing algorithm for other tasks like text classification

**Unsmoothed** maximum likelihood estimate of the unigram probability of the word $w_i$: its count $c_i$ normalized by the total number of word tokens $N$
$$
P\left(w_{i}\right)=\frac{c_{i}}{N}
$$
**Laplace smoothed**: 

- Merely adds one to each count
- Since there are $V$ words in the vocabulary and each one was incremented, we also need to adjust the denominator to take into account the extra $V$ observations.

$$
P_{\text {Laplace}}\left(w_{i}\right)=\frac{c_{i}+1}{N+V}
$$

**Adjust count** $c^*$
$$
c_{i}^{*}=\left(c_{i}+1\right) \frac{N}{N+V}
$$

- easier to compare directly with the MLE counts and can be turned into a probability like an MLE count by normalizing by $N$

  > $\frac{c_I^*}{N} = \left(c_{i}+1\right) \frac{N}{N+V} \cdot \frac{1}{N} =\frac{c_{i}+1}{N+V} = P_{\text {Laplace}}\left(w_{i}\right)$

#### Another aspect of smoothing

A related way to view smoothing is as **discounting** (lowering) some non-zero counts in order to get the probability mass that will be assigned to the zero counts. 

Thus, instead of referring to the discounted counts $c^*$, we might describe a smoothing algorithm in terms of a relative **discount** $d_c$ 
$$
d_c = \frac{c^*}{c}
$$
(the ratio of the discounted counts to the original counts)

#### Example

Smooth the Berkeley Restaurant Project bigrams

- Original(unsmoothed) bigram counts and probabilities

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-02%2016.56.21.png" alt="截屏2020-06-02 16.56.21" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-02%2016.58.56.png" alt="截屏2020-06-02 16.58.56" style="zoom:80%;" />

- Add-one smoothed counts and probabilities

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-03%2016.33.50.png" alt="截屏2020-06-03 16.33.50" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-03%2016.36.08.png" alt="截屏2020-06-03 16.36.08" style="zoom:80%;" />

Computation:

- Recall: normal bigram probabilities are computed by normalizing each row of counts by the unigram count
  $$
  P\left(w_{n} | w_{n-1}\right)=\frac{C\left(w_{n-1} w_{n}\right)}{C\left(w_{n-1}\right)}
  $$

- add-one smoothed bigram: augment the unigram count by the number of total word types in the vocabulary

  $$
  P_{\text {Laplace }}^{*}\left(w_{n} | w_{n-1}\right)=\frac{C\left(w_{n-1} w_{n}\right)+1}{\sum_{w}\left(C\left(w_{n-1} w\right)+1\right)}=\frac{C\left(w_{n-1} w_{n}\right)+1}{C\left(w_{n-1}\right)+V}
  $$

- It is often convenient to reconstruct the count matrix so we can see how much a smoothing algorithm has changed the original counts.
  $$
  c^{*}\left(w_{n-1} w_{n}\right)=\frac{\left[C\left(w_{n-1} w_{n}\right)+1\right] \times C\left(w_{n-1}\right)}{C\left(w_{n-1}\right)+V}
  $$
  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-03%2017.19.19.png" alt="截屏2020-06-03 17.19.19" style="zoom:80%;" />

Add-one smoothing has made a very big change to the counts.

- $C(\text{want to})$ changed from 609 to 238
- $P(to|want)$ decreases from .66 in the unsmoothed case to .26 in the smoothed case
- The discount $d$ (the ratio between new and old counts) shows us how strikingly the counts for each prefix word have been reduced
  - the discount for the bigram *want to* is .39
  - the discount for *Chinese food* is .10

The sharp change in counts and probabilities occurs because too much probability mass is moved to all the zeros.

### Add-k smoothing

Instead of adding 1 to each count, we add a fractional count $k$
$$
P_{\mathrm{Add}-\mathrm{k}}^{*}\left(w_{n} | w_{n-1}\right)=\frac{C\left(w_{n-1} w_{n}\right)+k}{C\left(w_{n-1}\right)+k V}
$$

- $k$: can be chosen by optimizing on a devset (validation set)

Add-k smoothing

- useful for some tasks (including text classification)
- still doesn’t work well for language modeling, generating counts with poor variances and often inappropriate discounts 🤪

### Backoff and interpolation

Sometimes using less context is a good thing, helping to generalize more for contexts that the model hasn’t learned much about.

#### Backoff

**💡 “Back off” to a lower-order n-gram if we have zero evidence for a higher-order n-gram**

- If the n-gram we need has zero counts, we approximate it by backing off to the (n-1)-gram. We continue backing off until we reach a history that has some counts.
  - *we use the trigram if the evidence is sufficient, otherwise we use the bigram, otherwise the unigram.*

##### Katz backoff

- Rely on a discounted probability $P^*$ if we’ve seen this n-gram before (i.e., if we have non-zero counts)

  - We have to discount the higher-order n-grams to save some probability mass for the lower order n-grams

    > if the higher-order n-grams aren’t discounted and we just used the undiscounted MLE probability, then as soon as we replaced an n-gram which has zero probability with a lower-order n-gram, we would be adding probability mass, and the total probability assigned to all possible strings by the language model would be greater than 1!

- Otherwise, we recursively back off to the Katz probability for the shorter-history (n-1)-gram.

$\Rightarrow$ The probability for a backoff n-gram $P_{\text{BO}}$ is
$$
P_{\mathrm{BO}}\left(w_{n} | w_{n-N+1}^{n-1}\right)=\left\{\begin{array}{ll}
P^{*}\left(w_{n} | w_{n-N+1}^{n-1}\right), & \text { if } C\left(w_{n-N+1}^{n}\right)>0 \\
\alpha\left(w_{n-N+1}^{n-1}\right) P_{\mathrm{BO}}\left(w_{n} | w_{n-N+2}^{n-1}\right), & \text { otherwise }
\end{array}\right.
$$

- $P^*$: discounted probability
- $\alpha$: a function to distribute the discounted probability mass to the lower order n-grams



#### Interpolation

💡 Mix the probability estimates from all the n-gram estimators, weighing and combining the trigram, bigram, and unigram counts.

In *simple* **linear** interpolation, we combine different order n-grams by linearly interpolating all the models. I.e., we estimate the trigram probability $P\left(w_{n} | w_{n-2} w_{n-1}\right)$ by mixing together the unigram, bigram, and trigram probabilities, each weighted by a $\lambda$
$$
\begin{array}{ll}
\hat{P}\left(w_{n} | w_{n-2} w_{n-1}\right) = & \lambda_{1} P\left(w_{n} | w_{n-2} w_{n-1}\right) \\
&+ \lambda_{2} P\left(w_{n} | w_{n-1}\right) \\
&+ \lambda_{3} P\left(w_{n}\right)
\end{array}
$$
s.t.
$$
\sum_{i} \lambda_{i}=1
$$
In a *slightly more sophisticated* version of linear interpolation, each $\lambda$ weight is computed by conditioning on the context. 

- If we have particularly accurate counts for a particular bigram, we assume that the counts of the trigrams based on this bigram will be more trustworthy, so we can make the $λ$s for those trigrams higher and thus give that trigram more weight in the interpolation.

$$
\begin{array}{ll}
\hat{P}\left(w_{n} | w_{n-2} w_{n-1}\right)=& \lambda_{1}\left(w_{n-2}^{n-1}\right) P\left(w_{n} | w_{n-2} w_{n-1}\right) \\
&+\lambda_{2}\left(w_{n-2}^{n-1}\right) P\left(w_{n} | w_{n-1}\right) \\
&+\lambda_{3}\left(w_{n-2}^{n-1}\right) P\left(w_{n}\right)
\end{array}
$$

##### How to set $\lambda$s?

Learn from a **held-out** corpus

- Held-out corpus: an additional training corpus that we use to set hyperparameters like these $λ$ values, by choosing the $λ$ values that maximize the likelihood of the held-out corpus.
- We fix the n-gram probabilities and then search for the $λ$ values that give us the highest probability of the held-out set
  - Common method: **EM** algorithm

### Kneser-Ney Smoothing 👍

One of the most commonly used and best performing n-gram smoothing methods 👏

Based on **absolute discounting**

- subtracting a fixed (absolute) discount $d$ from each count.
- 💡 Intuition: 
  - since we have good estimates already for the very high counts, a small discount *d* won’t affect them much
  - It will mainly modify the smaller counts, for which we don’t necessarily trust the estimate anyway

![截屏2020-06-03 22.51.39](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-03%2022.51.39.png)

Except for the held-out counts for 0 and 1, all the other bigram counts in the held-out set could be estimated pretty well by just subtracting 0.75 from the count in the training set! In practice this discount is actually a good one for bigrams with counts 2 through 9.

The equation for interpolated absolute discounting applied to bigrams:
$$
P_{\text {AbsoluteDiscounting }}\left(w_{i} | w_{i-1}\right)=\frac{C\left(w_{i-1} w_{i}\right)-d}{\sum_{v} C\left(w_{i-1} v\right)}+\lambda\left(w_{i-1}\right) P\left(w_{i}\right)
$$

- First term: discounted bigram
  - We could just set all the $d$ values to .75, or we could keep a separate discount value of 0.5 for the bigrams with counts of 1.
- Second term: unigram with an interpolation weight $\lambda$

**Kneser-Ney discounting** augments absolute discounting with a more sophisticated way to handle the lower-order unigram distribution.

Sophisticated means: Instead of $P(w)$ which answers the question “How likely is $w$?”, we’d like to create a unigram model that we might call $P_{\text{CONTINUATION}}$, which answers the question “How likely is *w* to appear as a novel continuation?”

How can we estimate this probability of seeing the word $w$ as a novel continuation, in a new unseen context?

💡 The Kneser-Ney intuition: base our $P_{\text{CONTINUATION}}$ on the *number of different contexts word* $w$ *has appeared in* (the number of bigram types it completes).

- Every bigram type was a novel continuation the first time it was seen.
- We hypothesize that words that have appeared in more contexts in the past are more likely to appear in some new context as well.

The number of times a word $w$ appears as a novel continuation can be expressed as:
$$
P_{\mathrm{CONTINUATION}}(w) \propto|\{v: C(v w)>0\}|
$$
To turn this count into a probability, we normalize by the total number of word bigram types:
$$
P_{\mathrm{CONTINUATION}}(w)=\frac{|\{v: C(v w)>0\}|}{\left|\left\{\left(u^{\prime}, w^{\prime}\right): C\left(u^{\prime} w^{\prime}\right)>0\right\}\right|}
$$
An equivalent formulation based on a different metaphor is to use the number of word types seen to precede $w$, normalized by the number of words preceding all words,
$$
P_{\mathrm{CONTINUATION}}(w)=\frac{|\{v: C(v w)>0\}|}{\sum_{w^{\prime}}\left|\left\{v: C\left(v w^{\prime}\right)>0\right\}\right|}
$$
The final equation for **Interpolated Kneser-Ney smoothing** for bigrams is:
$$
P_{\mathrm{KN}}\left(w_{i} | w_{i-1}\right)=\frac{\max \left(C\left(w_{i-1} w_{i}\right)-d, 0\right)}{C\left(w_{i-1}\right)}+\lambda\left(w_{i-1}\right) P_{\mathrm{CONTINUATION}}\left(w_{i}\right)
$$

- $λ$: normalizing constant that is used to distribute the probability mass
  $$
  \lambda\left(w_{i-1}\right)=\frac{d}{\sum_{v} C\left(w_{i-1} v\right)}\left|\left\{w: C\left(w_{i-1} w\right)>0\right\}\right|
  $$

  - First term: normalized discount
  - Second term: the number of word types that can follow $w_{i-1}$. or, equivalently, the number of word types that we discounted (i.e., the number of times we applied the normalized discount.)

The general recursive formulation is 
$$
P_{\mathrm{KN}}\left(w_{i} | w_{i-n+1}^{i-1}\right)=\frac{\max \left(c_{K N}\left(w_{i-n+1}^{i}\right)-d, 0\right)}{\sum_{v} c_{K N}\left(w_{i-n+1}^{i-1} v\right)}+\lambda\left(w_{i-n+1}^{i-1}\right) P_{K N}\left(w_{i} | w_{i-n+2}^{i-1}\right)
$$

- $C_{KN}$: depends on whether we are counting the highest-order n-gram being interpolated or one of the lower-order n-grams

  $c_{K N}(\cdot)=\left\{\begin{array}{l}\text { count }(\cdot) \quad \text { for the highest order } \\ \text { continuationcount }(\cdot) \quad \text { for lower orders }\end{array}\right.$
  
  - $\operatorname{continuationcount}(\cdot)$: the number of unique single word contexts for $\cdot$

At the termination of the recursion, unigrams are interpolated with the uniform distribution
$$
P_{\mathrm{KN}}(w)=\frac{\max \left(c_{K N}(w)-d, 0\right)}{\sum_{w^{\prime}} c_{K N}\left(w^{\prime}\right)}+\lambda(\varepsilon) \frac{1}{V}
$$

- $\varepsilon$: empty string



## Perplexity’s Relation to Entropy

Recall: A better n-gram model is one that assigns a higher probability to the test data, and perplexity is a normalized version of the probability of the test set.

**Entropy**: a measure of information

- Given:

  - A random variable $X$ ranging over whatever we are predicting (words, letters, parts of speech, the set of which we’ll call $χ$)
  - with a particular probability function $p(x)$

- The entropy of the random variable $X$ is
  $$
  H(X)=-\sum_{x \in \chi} p(x) \log _{2} p(x)
  $$

  - If we use log base 2, the resulting value of entropy will be measured in **bits**.

💡 Intuitive way to think about entropy: a **lower bound** on the number of bits it would take to encode a certain decision or piece of information in the optimal coding scheme.

------

**Example**

Imagine that we want to place a bet on a horse race but it is too far to go all the way to Yonkers Racetrack, so we’d like to send a short message to the bookie to tell him which of the eight horses to bet on.

One way to encode this message is just to use the binary representation of the horse’s number as the code: horse 1 would be `001`, horse 2 `010`, horse 3 `011`, and so on, with horse 8 coded as `000`. On average we would be sending 3 bits per race.

Suppose that the spread is the actual distribution of the bets placed and that we represent it as the prior probability of each horse as follows:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-06-04%2010.23.19.png" alt="截屏2020-06-04 10.23.19" style="zoom:80%;" />

The entropy of the random variable *X* that ranges over horses gives us a lower bound on the number of bits and is
$$
\begin{aligned}
H(X) &=-\sum_{i=1}^{i=8} p(i) \log p(i) \\
&=-\frac{1}{2} \log \frac{1}{2}-\frac{1}{4} \log \frac{1}{4}-\frac{1}{8} \log \frac{1}{8}-\frac{1}{16} \log \frac{1}{16}-4\left(\frac{1}{64} \log \frac{1}{64}\right) \\
&=2 \text { bits }
\end{aligned}
$$
A code that averages 2 bits per race can be built with *short* encodings for *more probable* horses, and *longer* encodings for *less probable* horses. E.g. we could encode the most likely horse with the code `0`, and the remaining horses as `10`, then `110`, `1110`, `111100`, `111101`, `111110`, and `111111`.

Suppose horses are equally likely. In this case each horse would have a probability of $\frac{1}{8}$. The entropy is then
$$
H(X)=-\sum_{i=1}^{i=8} \frac{1}{8} \log \frac{1}{8}=-\log \frac{1}{8}=3 \mathrm{bits}
$$

------

Most of what we will use entropy for involves ***sequences***.

For a grammar, for example, we will be computing the entropy of some sequence of words $W=\{w_0, w_1, w_2, \dots, w_n\}$. One way to do this is to have a variable that ranges over sequences of words. For example we can compute the entropy of a random variable that ranges over all ***finite*** sequences of words of length $n$ in some language $L$
$$
H\left(w_{1}, w_{2}, \ldots, w_{n}\right)=-\sum_{W_{1}^{n} \in L} p\left(W_{1}^{n}\right) \log p\left(W_{1}^{n}\right)
$$
**Entropy rate** (**per-word entropy**): entropy of this sequence divided by the number of word
$$
\frac{1}{n} H\left(W_{1}^{n}\right)=-\frac{1}{n} \sum_{W_{1}^{n} \in L} p\left(W_{1}^{n}\right) \log p\left(W_{1}^{n}\right)
$$
For sequence $L$ of ***infinite*** length, the entropy rate $H(L)$ is
$$
\begin{aligned}
H(L) &=\lim _{n \rightarrow \infty} \frac{1}{n} H\left(w_{1}, w_{2}, \ldots, w_{n}\right) \\
&=-\lim _{n \rightarrow \infty} \frac{1}{n} \sum_{W \in L} p\left(w_{1}, \ldots, w_{n}\right) \log p\left(w_{1}, \ldots, w_{n}\right)
\end{aligned}
$$

### The Shannon-McMillan-Breiman theorem

If the language is regular in certain ways (to be exact, if it is both **stationary** and **ergodic**), then
$$
H(L)=\lim _{n \rightarrow \infty}-\frac{1}{n} \log p\left(w_{1} w_{2} \ldots w_{n}\right)
$$
I.e., we can take a single sequence that is long enough instead of summing over all possible sequences. 

- 💡 Intuition: a long-enough sequence of words will contain in it many other shorter sequences and that each of these shorter sequences will reoccur in the longer sequence according to their probabilities.

**Stationary**

A stochastic process is said to be **stationary** if the probabilities it assigns to a sequence are *invariant* with respect to shifts in the time index.

- I.e., the probability distribution for words at time $t$ is the same as the probability distribution at time $t+1$.
- Markov models, and hence n-grams, are stationary.
  - E.g.,  in bigram, $P_i$ is dependent only on $P_{i-1}$. If we shift our time index by $x$, $P_{i+x}$ is still dependent on  $P_{i+x-1}$

- Natural language is NOT stationary
  - the probability of upcoming words can be dependent on events that were arbitrarily distant and time dependent. 

To summarize, by making some incorrect but convenient simplifying assumptions, **we can compute the entropy of some stochastic process by taking a very long sample of the output and computing its average log probability.**

### Cross-entropy

Useful when we don’t know the actual probability distribution $p$ that generated some data

It allows us to use some $m$, which is a model of $p$ (i.e., an approximation to $p$). The

cross-entropy of $m$ on $p$ is defined by
$$
H(p, m)=\lim _{n \rightarrow \infty}-\frac{1}{n} \sum_{W \in L} p\left(w_{1}, \ldots, w_{n}\right) \log m\left(w_{1}, \ldots, w_{n}\right)
$$
(we draw sequences according to the probability distribution $p$, but sum the log of their probabilities according to $m$)

Following the Shannon-McMillan-Breiman theorem, for a stationary ergodic process: 
$$
H(p, m)=\lim _{n \rightarrow \infty}-\frac{1}{n} \log m\left(w_{1} w_{2} \ldots w_{n}\right)
$$
(as for entropy, we can estimate the cross-entropy of a model $m$ on some distribution $p$ by taking a single sequence that is long enough instead of summing over all possible sequences)

The cross-entropy $H(p. m)$ is an **upper bound** on the entropy $H(p)$:
$$
H(p)\leq H(p, m)
$$
This means that we can use some simplified model $m$ to help estimate the true entropy of a sequence of symbols drawn according to probability $p$

- The more accurate $m$ is, the closer the cross-entropy $H(p, m)$ will be to the true entropy $H(p)$
  - Difference between $H(p, m)$ and $H(p)$ is a measure of how accurate a model is
  - The more accurate model will be the one with the lower cross-entropy. 

#### Relationship between perplexity and cross-entropy

Cross-entropy is defined in the limit, as the length of the observed word sequence goes to infinity. We will need an approximation to cross-entropy, relying on a (sufficiently long) sequence of fixed length. 

This approximation to the cross-entropy of a model $M=P\left(w_{i} | w_{i-N+1} \dots w_{i-1}\right)$ on a sequence of words $W$ is
$$
H(W)=-\frac{1}{N} \log P\left(w_{1} w_{2} \ldots w_{N}\right)
$$
The **perplexity** of a model $P$ on a sequence of words $W$ is now formally defined as

the exp of this cross-entropy:
$$
\begin{aligned}
\operatorname{Perplexity}(W) &=2^{H(W)} \\
&=P\left(w_{1} w_{2} \ldots w_{N}\right)^{-\frac{1}{N}} \\
&=\sqrt[N]{\frac{1}{P\left(w_{1} w_{2} \ldots w_{N}\right)}} \\
&=\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P\left(w_{i} | w_{1} \ldots w_{i-1}\right)}}
\end{aligned}
$$
