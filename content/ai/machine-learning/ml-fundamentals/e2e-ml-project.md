---
# Title, summary, and position in the list
linktitle: "e2e ML Project"
summary: ""
weight: 120

# Basic metadata
title: "End-to-End Machine Learning Project"
date: 2020-08-17
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Machine Learning", "ML Basics"]
categories: ["Machine Learning"]
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
    machine-learning:
        parent: ml-fundamentals
        weight: 2

---



![e2e_ML_Project](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/e2e_ML_Project.png)

## 1. Look at the big picture

### 1.1 Frame the problem

Consider the business objective: How do we expect to use and benefit from this model?

### 1.2 Select a performance measure

### 1.3 Check the assumptions

List and verify the assumptions.



## 2. Get the data

### 2.1 Download the data

Automate this process: Create a small function to handle downloading, extracting, and storing data.

### 2.2 Take a quick look at the data

- Use `pandas.head()` to look at the top rows of the data
- Use `pandas.info()` to get a quick description of the data
  - For categorical attributes, use `value_counts()` to see categories and the #samples of each category
  - For numerical attributes, use `describe()` to get a summary of the numerical attributes.

### Create a test set

- If dataset is large enough, use **purely random sampling**. (`train_test_split`)

- If the test set need to be representative of the overall data, use **stratified sampling**.

  

## 3. Discover and visualize the data to gain insights

1. Make sure put the test set aside and only explore the training set
2. If the trainingset is very large, sample an exploration set to make manipulations easy and fast

### 3.1 Visualizing data

### 3.2 Look for correlations

Two ways:

- Compute the **standard correlation coefficient** (also called **Pearson's r**) between every pair of attributes using the `corr()` method.
- Or use `panda`'s `scatter_matrix` function

### 3.3 Experimenting with attribute combinations



## 4. Prepare the data for ML algorithms

**Firstly, ensure a clean training set and separate the predictors and labels.**

### 4.1 Data cleaning

Handle missing features: 

- Get rid of the corresponding samples (districts) -> use `dropna()`
- Get rid of the whole attribute -> use `drop()`
- Set the values to some value (zero, the mean, the median, etc.) -> use `fillna()`

Or apply `SimpleImputer` from Scikit-Learn to all the numerical attributes.

### 4.2 Handle text and categorical attributes

Most ML algorithms prefer to work with numbers anyway.
Transform text and categorical attributes to numerical attributes Using One-hot encoding.

### 4.3 Custom transformers

The custom transformer should work seamlessly with Scikit-Learn functionalities (such as pipelines).
-> Create a class and implement three methods:

- `fit()`
- `transform()`
- `fit_transform()` (can get it by simply adding `TransfromerMixin` as a base class)

If we add `BaseEstimator` as a bass class, we can get two extra methods 

- `get_params()`
- `set_params()`
  that will be useful for automatic hyperparameter tuning.

### 4.4 Feature scaling

Comman ways:

- **Min-max scaling (normalization)**: Use `MinMaxScalar`
- **Standardization**
  - Use `StandardScalar`
  - Less affected by outliners

### 4.5 Transformation pipelines

Group sequences of transformations into one step.

`Pipeline` from `scikit-learn`:

- a list of name/estimator pairs defining a sequence of steps
- the last estimator must be transformers (must have a `fit_transform()` method)
- names can be anything but must be unique and don't contain double underscores "__"

More convenient is to use a **single** transformer to handle the categorical columns and the numerical columns.
-> Use `ColumbTransformer`: handle all columns, applying the appropriate transformations to each column and also works great with Pandas DataFrames.



## 5. Select a model and train it

### 5.1 Train and evaluate on the trainging set

### 5.2 Better evaluation using Cross-Validation



## 6. Fine-tune the model

### 6.1 Grid search

When exploring **relatively few** combinations, use `GridSearchCV`: Tell it which hyperparameters we want to experiment with, and what values to try out. Then it will evaluate all the possible combinations of hyperparameter values, using cross-validation.

### 6.2 Randomized search

When the hyperparameter search space is **large**, use `RandomizedSearchCV`. It evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.

### 6.3 Ensemble methods

Try to combine the models that perform best. 

### 6.4 Analyze the best models and their errors

Gain good insights on the problem by inspecting the best models.

### 6.5 Evaluate the system on the test set

1. Get the predictors and labels from test set

2. Run full pipeline to transform the data 

3. Evaluate the final model on the test set

    

## 7. Present the solution



## 8. Launch, monitor, and maintain the system

- Plug the production input data source into the system and write test
- Write monitoring code to check system's live performance at regular intervals and trigger alerts when it drops
- Evaluate the system's input data quality
- Train the models on a regular basis using fresh data (automate this precess as much as possible!)