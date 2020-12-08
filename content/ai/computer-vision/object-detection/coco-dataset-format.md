---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 1005

# Basic metadata
title: "COCO JSON Format for Object Detection"
date: 2020-12-02
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Object Detection", "COCO"]
categories: ["Computer Vision"]
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
    computer-vision:
        parent: object-detection
        weight: 5
---

The COCO dataset is formatted in [JSON](https://www.w3schools.com/js/js_json_syntax.asp) and is a collection of “info”, “licenses”, “images”, “annotations”, “categories” (in most cases), and “segment info” (in one case).

```json
{
    "info": {...},
    "licenses": [...],
    "images": [...],
    "annotations": [...],
    "categories": [...], <-- Not in Captions annotations
    "segment_info": [...] <-- Only in Panoptic annotations
}
```

Note:

- `categories` field is NOT in Captions annotations
- `segment_info` field is ONLY in Panoptic annotations

## Info

The “info” section contains **high level information** about the dataset. If you are creating your own dataset, you can fill in whatever is appropriate.

Example:

```json
"info": {
    "description": "COCO 2017 Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2017,
    "contributor": "COCO Consortium",
    "date_created": "2017/09/01"
}
```

## Lincenses

The “licenses” section contains a **list** of image licenses that apply to images in the dataset

Example:

```json
"licenses": [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    },
    {
        "url": "http://creativecommons.org/licenses/by-nc/2.0/",
        "id": 2,
        "name": "Attribution-NonCommercial License"
    },
    ...
]
```

## Images

- Contains the **complete list** of images in your dataset
- No labels, bounding boxes, or segmentations specified in this part, it's simply a list of images and information about each one. 
- `coco_url`, `flickr_url`, and `date_captured` are just for reference. Your deep learning application probably will only need the **`file_name`**.
-  Image ids need to be **unique** (among other images)
  - They do not necessarily need to match the file name (unless the deep learning code you are using makes an assumption that they’ll be the same)

Example:

```json
"images": [
    {
        "license": 4,
        "file_name": "000000397133.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": 427,
        "width": 640,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": 397133
    },
    {
        "license": 1,
        "file_name": "000000037777.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "height": 230,
        "width": 352,
        "date_captured": "2013-11-14 20:55:31",
        "flickr_url": "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg",
        "id": 37777
    },
    ...
]
```

## Annotations

COCO has five annotation types: for [object detection](http://cocodataset.org/#detection-2018), [keypoint detection](http://cocodataset.org/#keypoints-2018), [stuff segmentation](http://cocodataset.org/#stuff-2018), [panoptic segmentation](http://cocodataset.org/#panoptic-2018), and [image captioning](http://cocodataset.org/#captions-2015). The annotations are stored using [JSON](http://json.org/).

### Object detection

it draws shapes around objects in an image. It has a list of **categories** and **annotations**.

#### Categories

- Contains a list of **categories** (e.g. dog, boat) 
  - each of those belongs to a **supercategory** (e.g. animal, vehicle). 
- The original COCO dataset contains 90 categories. 
- You can use the existing COCO categories or create an entirely new list of your own. 
- **Each category id must be unique (among the rest of the categories).**

Example:

```json
"categories": [
    {
        "supercategory": "person",
        "id": 1,
        "name": "person"
    },
    {
        "supercategory": "vehicle",
        "id": 2,
        "name": "bicycle"
    },
    {
        "supercategory": "vehicle",
        "id": 3,
        "name": "car"
    },
    ...
]
```

#### Annotations

- `segmentation` : list of points (represented as $(x, y)$ coordinate  ) which define the shape of the object

- `area` : measured in pixels (e.g. a 10px by 20px box would have an area of 200)
- `iscrowd` : specifies whether the segmentation is for a single object (`iscrowd=0`) or for a group/cluster of objects (`iscrowd=1`)
- `image_id`: corresponds to a specific image in the dataset
- `bbox` : bounding box, format is `[top left x position, top left y position, width, height]` 
- `category_id`: corresponds to a single category specified in the categories section
- `id`: Each annotation also has an id (unique to all other annotations in the dataset)

Example:

```json
"annotations": [
    {
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],
        "area": 702.1057499999998,
        "iscrowd": 0,
        "image_id": 289343,
        "bbox": [473.07,395.93,38.65,28.67],
        "category_id": 18,
        "id": 1768
    },
    ...
]
```

- Has a segmentation list of vertices (x, y pixel positions)
- Has an area of 702 pixels (pretty small) and a bounding box of [473.07,395.93,38.65,28.67]
- Is not a crowd (meaning it’s a single object)
- Is category id of 18 (which is a dog)
- Corresponds with an image with id 289343 (which is a person on a strange bicycle and a tiny dog)

## Example

Source: https://roboflow.com/formats/coco-json

```json
{
    "info": {
        "year": "2020",
        "version": "1",
        "description": "Exported from roboflow.ai",
        "contributor": "Roboflow",
        "url": "https://app.roboflow.ai/datasets/hard-hat-sample/1",
        "date_created": "2000-01-01T00:00:00+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "name": "Public Domain"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "Workers",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "head",
            "supercategory": "Workers"
        },
        {
            "id": 2,
            "name": "helmet",
            "supercategory": "Workers"
        },
        {
            "id": 3,
            "name": "person",
            "supercategory": "Workers"
        }
    ],
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "0001.jpg",
            "height": 275,
            "width": 490,
            "date_captured": "2020-07-20T19:39:26+00:00"
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                45,
                2,
                85,
                85
            ],
            "area": 7225,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 1,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                324,
                29,
                72,
                81
            ],
            "area": 5832,
            "segmentation": [],
            "iscrowd": 0
        }
    ]
}
```



## Reference

- [COCO Data format](https://cocodataset.org/#format-data)

- [Create COCO Annotations From Scratch](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format)

  - Video tutorial

    {{< youtube h6s61a_pqfM >}}