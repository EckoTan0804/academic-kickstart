---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 502

# Basic metadata
title: "Plotly"
date: 2020-08-31
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Visualization", "Matploblib"]
categories: ["coding"]
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
    python:
        parent: visualization
        weight: 2

---





## Dash

![Numfocus - Plotly Dash Logo - 1176x528 PNG Download - PNGkit](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/images.png)

[Dash](https://plot.ly/products/dash), a member of Plotly’s open-source tools, is a Open Source Python library for creating **reactive, Web-based** applications. Dash is a user interface library for creating analytical web applications. Those who use Python for data analysis, data exploration, visualization, modelling, instrument control, and reporting will find immediate use for Dash.

### Why is Dash good?

- Dash app code is **declarative** and **reactive**, which makes it easy to build complex apps that contain many interactive elements.
- Every aesthetic element of the app is customizable
- While Dash apps are viewed in the web browser, you do NOT have to write any Javascript or HTML. Dash provides a Python interface to a rich set of interactive web-based components.
- Dash provides a simple reactive decorator for binding your custom data analysis code to your Dash user interface.
- Through these two abstractions — Python components and reactive functional decorators — Dash abstracts away all of the technologies and protocols that are required to build an interactive web-based application.

Dash is simple enough that you can bind a user interface around your Python code in an afternoon. :clap:

### Architecture

#### Frontend and backend

Dash leverages the power of Flask and React, putting them to work for Python data scientists who may not be expert Web programmers.

- Dash applications are web servers running **[Flask](http://flask.pocoo.org/)** and communicating **JSON** packets over HTTP requests.
- Dash’s frontend renders components using **React.js**
  - Dash components are Python classes that encode the properties and values of a specific React component and that serialize as JSON.
  - The full set of HTML tags, like `<div/>`, `<img/>`, `<table/>` are also rendered dynamically with React and their Python classes are available through the `dash_html_component` library.

#### CSS

CSS and default styles are kept out of the core library for modularity, independent versioning, and to encourage Dash App developers to customize the look-and-feel of their apps.

#### Data Visualization

Dash ships with a Graph component that renders charts with [plotly.js](https://github.com/plotly/plotly.js).

- built on top of D3.js (for publication-quality, vectorized image export) and WebGL (for high performance visualization)

- declarative, open source, fast
- supports a complete range of scientific, financial, and business charts.
- Dash’s Graph element shares the same syntax as the open-source [plotly.py](https://plot.ly/python) library, so you can easily to switch between the two.

#### Open Source Repositories

- Dash backend: https://github.com/plotly/dash
- Dash frontend: https://github.com/plotly/dash-renderer
- Dash core component library: https://github.com/plotly/dash-core-components
- Dash HTML component library: https://github.com/plotly/dash-html-components
- Dash component archetype (React-to-Dash toolchain): https://github.com/plotly/dash-components-archetype
- Dash docs and user guide: https://github.com/plotly/dash-docs, hosted at https://plot.ly/dash
- Plotly.js — the graphing library used by Dash: https://github.com/plotly/plotly.js

### Getting started

- [Dash User Guide](https://dash.plotly.com/)

## Deployment

If you want to share public Dash apps for free, you can deploy it on [Heroku](www.heroku.com), one of the easiest platforms for deploying and managing public Flask applications.

Check out the tutorial: [Deploying Dash Apps](https://dash.plotly.com/deployment)

### Heroku

**Heroku** is a cloud [platform as a service](https://en.wikipedia.org/wiki/Platform_as_a_service) (PaaS) supporting several [programming languages](https://en.wikipedia.org/wiki/Programming_language). It supports supports [Java](https://en.wikipedia.org/wiki/Java_(programming_language)), [Node.js](https://en.wikipedia.org/wiki/Node.js), [Scala](https://en.wikipedia.org/wiki/Scala_(programming_language)), [Clojure](https://en.wikipedia.org/wiki/Clojure), [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), [PHP](https://en.wikipedia.org/wiki/PHP), and [Go](https://en.wikipedia.org/wiki/Go_(programming_language)). For this reason, Heroku is said to be a [polyglot platform](https://en.wikipedia.org/wiki/Polyglot_(computing)) as it has features for a [developer](https://en.wikipedia.org/wiki/Software_developer) to build, run and scale applications in a similar manner across most languages.

## Reference

- [🌟 Introducing Dash 🌟](https://medium.com/plotly/introducing-dash-5ecf7191b503)

