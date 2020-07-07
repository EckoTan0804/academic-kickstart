---
# Basic info
title: "Customizing the Order of Left Menu Items"
date: 2020-07-07
draft: false
type: docs # page type
authors: ["admin"]
tags: ["hugo-academic"]
categories: ["Blog customization"]
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
  caption: ""
  image: ""
---

If we follow the [official tutorials of Academic](https://sourcethemes.com/academic/docs/managing-content/#create-a-course-or-documentation) to create a course or documentation, the default order of the menu items in the left side-menu is **alphabetical**. However, it can make more sense if we could customize the order.

Let's take [online course demo](https://academic-demo.netlify.app/courses/) as an example. (Take a look at its code in [Github repo](https://github.com/gcushen/hugo-academic/tree/master/exampleSite/content/courses).)

Originally, the left side-menu looks like this:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-07%2012.11.01.png" alt="截屏2020-07-07 12.11.01" style="zoom:50%;" />

We want to customize the menu like below:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-07-07%2012.11.48.png" alt="截屏2020-07-07 12.11.48" style="zoom:50%;" />

How can we do that?



## Step by Step Customization

According to [Ordering of Tutorial Contents #831](https://github.com/gcushen/hugo-academic/issues/831), we could do it as follows:

### 1.Define parent menu items in `config/_default/menus.toml`

Add the following codes in `menu.toml`:

```toml
################################
# Courses
################################
[[example]]
  name = "Example Topic"
  weight = 10
  identifier = "example-topic"

[[example]]
  name = "Another Topic"
  weight = 20
  identifier = "another-topic"
```

Notice that `example` is the folder name. If you rename the folder, you have to change `example` to `<newFolderName>`. (More see: [Menus](https://sourcethemes.com/academic/docs/managing-content/#menus))

### 2. Define parent menus items in `config/_default/config.toml`

Add the following codes in `config.toml`:

```toml
################################
# Courses
################################
[[menu.example]]
  name = "Example Topic"
  weight = 10
  identifier = "example-topic"

[[menu.example]]
  name = "Another Topic"
  weight = 20
  identifier = "another-topic"
```

### 3. Specify parent menu items in the front matter of each docs/tutorials page

In `example1.md`, we modify the `menu` parameter in front matter as followings:

```ymal
menu:
  example:
    parent: example-topic
    weight: 1
```

Note:

- For `parent` we use the identifier defined in step 2 instead of the parent menu item's name.
- `weight` specifies the position of this page under the parent item `example-topic`

We do the similar thing for `example2.md`:

```yaml
menu:
  example:
    parent: another-topic
    weight: 1
```

