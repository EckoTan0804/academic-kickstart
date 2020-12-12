---
# Basic info
title: "Data Science Project steps"
date: 2020-12-12
draft: false
# type: docs # page type
authors: ["admin"]
tags: ["Data Science"]
categories: ["Data Science"]
toc: true # Show table of contents?

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: ""
share: false  # Show social sharing links?
featured: true
lastmod: true

comments: false  # Show comments?
disable_comment: true
commentable: false  # Allow visitors to comment? Supported by the Page, Post, and Docs content types.

editable: false  # Allow visitors to edit the page? Supported by the Page, Post, and Docs content types.

# Optional header image (relative to `static/img/` folder).
header:
  image: ""
  caption: ""
  
---

## 7 Fundamental Steps to Complete a Data Analytics Project

Source: https://blog.dataiku.com/2019/07/04/fundamental-steps-data-project-success

![The seven steps of the Dataiku data process, from defining the goal to iteration](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/data_science_project_process.jpg)

1. **Understand the Business**

2. **Get data**

   1. Connect to database
   2. Use APIs
   3. Look for open data

3. **Explore and clean data (probably the longest, most annoying step)**

   - Make sure your project are compliant with data privacy regulations

4. **Enrich dataset**

   - Join all different sources and group logs to narrow data down to the essential features. 
     - Example: enrich data by creating time-based features
       - Extracting date components (month, hour, day of the week, week of the year, etc.)
       - Calculating differences between date columns
       - Flagging national holidays
   - Join datasets
     - Tools: Dataiku ([blend data through a simplified process](https://blog.dataiku.com/7-awesome-things-you-can-do-without-coding-in-dataiku))

   - Be extra careful NOT to insert unintended bias or other undesirable patterns into it

5. **Build helpful visualizations**

   - the best way to explore and communicate your findings and is the next phase of your data analytics project.

6. **Get predictive**

   - Use Machine Learning algorithms

7. **Iterate, Iterate, Iterate**

   -  constantly reevaluate, retrain it, and develop new features



## Data science process

Source: https://www.kdnuggets.com/2016/03/data-science-process.html

![img](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/525px-Data_visualization_process_v1.png)

1. **Frame the problem**

   The first thing you have to do before you solve a problem is to define exactly what it is. You need to be able to translate data questions into something actionable.

   It’s important that at the end of this stage, you have all of the information and context you need to solve this problem.

2. **Collect the raw data needed for your problem**
3. **Process the data for analysis**
   - Check for the common errors:
     - Missing values
     - Corrupted values
     - Timezone differences
     - Date range errors
   - Look through aggregates of your file rows and columns and sample some test values to see if your values make sense
   - If you detect something that doesn’t make sense, you’ll need to remove that data or replace it with a default value.
4. **Explore the data**
5. **Perform in-depth analysis**
6. **Communicate results of the analysis**



## Descriptive, Predictive & Prescriptive Analytics

Source: https://studyonline.unsw.edu.au/blog/descriptive-predictive-prescriptive-analytics

Businesses use analytics to explore and examine their data and then transform their findings into insights that ultimately help executives, managers and operational employees make better, more informed business decisions. Three key types of analytics businesses use are 

- **descriptive** analytics: *"what has happened in a business?"*
- **predictive** analytics: *"what could happen?"*
- **prescriptive** analytics: *"what should happen?"*

### Descriptive analytics

- A foundational starting point used to inform or prepare data for further analysis down the line :muscle:

- Commonly used form of data analysis whereby *historical* data is collected, organised and then presented in a way that is easily understood
- Focus only on what has *already happened* in a business 
- Unlike other methods of analysis, it is NOT used to draw inferences or predictions from its findings.
- Uses simple maths and statistical tools, such as arithmetic, averages and per cent changes
- Visual tools, such as line graphs and pie and bar charts, are used to present findings, meaning descriptive analytics can – and should – be easily understood by a wide business audience.

#### How does descriptive analytics work?

Descriptive analytics uses two key methods to discover historical data

- **Data aggregation** 
  - The process of collecting and organising data to create manageable data sets
  - These data sets are then used in the data mining phase where patterns, trends and meaning are identified and then presented in an understandable way
- **Data mining** (aka. data discovery)

#### **Five steps of descriptive analytics**

- Business metrics are decided
- The data required is identified
- The data is collected and prepared
- The data is analysed
- The data is presented

#### Examples of descriptive analytics

- Summarising past events such as sales and operations data or marketing campaigns
- Social media usage and engagement data such as Instagram or Facebook likes
- Reporting general trends
- Collating survey results

### Predictive analytics

- Focused on predicting and understanding what could happen in the future

#### How does predictive analytics work? 

- Based on probabilities

- Using a variety of techniques
  - data mining
  - statistical modelling (mathematical relationships between variables to predict outcomes)
  - machine learning algorithms (classification, regression and clustering techniques)
- attempts to forecast possible future outcomes and the likelihood of those events.

#### Advantages and disadvantages of predictive analysis

Predictive analytics can also [improve many areas of a business](https://www.sales-i.com/6-benefits-of-predictive-analytics), including:

- **Efficiency**, which could include inventory forecasting
- **Customer service**, which can help a company gain a better understanding of who their customers are and what they want in order to tailor recommendations
- **Fraud detection and prevention**, which can help companies identify patterns and changes
- **Risk reduction**, which, in the finance industry, might mean improved candidate screening 

However, this method of analysis relies on the existence of historical data, usually large amounts of it.

#### Examples of predictive analysis

- **E-commerce** – predicting customer preferences and recommending products to customers based on past purchases and search history
- **Sales** – predicting the likelihood that customers will purchase another product or leave the store
- **Human resources** – detecting if employees are thinking of quitting and then persuading them to stay
- **IT security** – identifying possible security breaches that require further investigation
- **Healthcare** – predicting staff and resource needs

### Prescriptive analytics

- Tells you what should be done
- The most complex stage of the business analytics process, requiring much more specialised analytics knowledge to perform, and for this reason it is rarely used in day-to-day business operations. 

- A number of techniques and tools – such as rules, statistics and machine learning algorithms – can be applied to available data, including both internal data (from within the business) and external data (such as data derived from social media).

#### What can prescriptive analytics tell us?

- Prescriptive analytics anticipates what, when and, importantly, why something might happen. 
- After considering the possible implications of each decision option, recommendations can then be made in regard to which decisions will best take advantage of future opportunities or mitigate future risks. 

#### Advantages and disadvantages of prescriptive analytics

- :thumbsup: Prescriptive analytics, when used effectively, provides invaluable insights in order to make the best possible, data-based decisions to optimise business performance.
- :thumbsdown: However, as with predictive analytics, this methodology requires large amounts of data to produce useful results, which isn’t always available. 
- :thumbsdown: machine learning algorithms, on which this analysis often relies, cannot always account for all external variables.

#### Examples of prescriptive analytics

- **Oil and manufacturing** – tracking fluctuating prices 
- **Manufacturing** – improving equipment management, maintenance, price modelling, production and storage
- **Healthcare** – improving patient care and healthcare administration by evaluating things such as rates of readmission and the cost-effectiveness of procedures
- **Insurance** – assessing risk in regard to pricing and premium information for clients
- **Pharmaceutical research** – identifying the best testing and patient groups for clinical trials.

## Top 5 biases to avoid in data science

Source: https://www.techrepublic.com/article/top-5-biases-to-avoid-in-data-science/

**Selection (or sample) bias**

It happens when the selected data is not representative of the cases the model will see. An all too frequent example of this is facial recognition trained predominantly on images of people with fair skin leading to algorithms that can't accurately identify people with darker skin.

**Confirmation bias**

This is where you toss out information that doesn't fit your preconceived notion--and it can be subconscious as all get out.

**Survivorship bias.**

This is where you select your data points because they're successful. Looking for data on what makes a product succeed? Don't just choose the successful products, you need data from the failures and the middle performers too.

**Availability bias**

This is where you use the data that's easy to get. A related bias is anchoring, where we give more importance to the first bit of data we get only because it was first.

**False causality**