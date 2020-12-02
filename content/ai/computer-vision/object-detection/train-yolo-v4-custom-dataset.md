---
# Title, summary, and position in the list
# linktitle: 
summary: ""
weight: 1004

# Basic metadata
title: "Train YOLO v4 on Custom Dataset"
date: 2020-11-04
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Computer Vision", "Object Detection", "YOLOv4"]
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
        weight: 4
---

## Clone and build Darknet

Clone darknet repo

```bash
git clone https://github.com/AlexeyAB/darknet
```

Change makefile to have GPU and OPENCV enabled

```bash
cd darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```

Verify CUDA

```bash
/usr/local/cuda/bin/nvcc --version
```

## Compile on Linux using `make`

Make darknet

```bash
make
```

- `GPU=1` : build with CUDA to accelerate by using GPU
- `CUDNN=1` : build with cuDNN v5-v7 to accelerate training by using GPU 
- `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
- `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
- `DEBUG=1` to bould debug version of Yolo
- `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU

{{% alert note %}}

Do not worry about any warnings when running `make` command.

{{% /alert %}}

## Prepare custom dataset

The custom dataset should be in **YOLOv4** or **darknet** format:

- For each `.jpg` image file, there should be a corresponding `.txt` file

  - In the same directory, with the same name, but with `.txt`-extension

    For example, if there's an `.jpg` image named `BloodImage_00001.jpg`, there should also be a corresponding `.txt` file named `BloodImage_00001.txt` 

- In this `.txt` file: object number and object coordinates on this image, for each object in new line. 

  Format:

  ```
  <object-class> <x_center> <y_center> <width> <height>
  ```

  - `<object-class>` : integer object number from `0` to `(classes-1)`
  - `<x_center> <y_center> <width> <height>` : float values **relative** to width and height of image, it can be equal from `(0.0 to 1.0]`
    - `<x_center> <y_center>` are center of rectangle (are not top-left corner)

## Configure files for training

0. For training `cfg/yolov4-custom.cfg` download the pre-trained weights-file [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) 

   ```bash
   cd darknet
   wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
   ```

1. In folder `./cfg`, create custom config file (let's call it `custom-yolov4-detector.cfg`) with the same content as in `yolov4-custom.cfg` and

   - change line **batch** to [`batch=64`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L3)

   - change line **subdivisions** to [`subdivisions=16`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)

   - change line **max_batches** to `classes*2000` but 

     - NOT less than number of training images
     - NOT less than number of training images
     - NOT less than 6000

     *e.g. `max_batches=6000` if you train for 3 classes*

   - change line **steps** to 80% and 90% of **max_batches** (*e.g. `steps=4800, 5400`*)

   - set network size `width=416 height=416` or any value multiple of 32

   - change line `classes=80` to number of objects in **each** of 3 `[yolo]`-layers

   - change [`filters=255`] to $ \text{filters}=(\text{classes} + 5) \times 3$ in the 3 `[convolutional]` before each `[yolo]` layer, keep in mind that it only has to be the last `[convolutional]` before each of the `[yolo]` layers.

     > Note: **Do not write in the cfg-file: `filters=(classes + 5) x 3`**!!!
     >
     > It has to be the specific number! 
     >
     > E.g. `classes=1` then should be `filters=18`; `classes=2` then should be `filters=21`
     >
     > So for example, for 2 objects, your custom config file should differ from `yolov4-custom.cfg` in such lines in **each** of **3** [yolo]-layers:
     >
     > ```
     > [convolutional]
     > filters=21
     > 
     > [region]
     > classes=2
     > ```

   - when using [`[Gaussian_yolo]`](https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L608) layers, change [`filters=57`] $ \text{filters}=(\text{classes} + 9) \times 3$ in the 3 `[convolutional]` before each `[Gaussian_yolo]` layer

2. Create file `obj.names` in the directory `data/`, with objects names - each in new line

3. Create fiel `obj.data` in the directory `data/`, containing (where **classes = number of objects**):

   For example, if we two objects

   ```
   classes = 2
   train  = data/train.txt
   valid  = data/test.txt
   names = data/obj.names
   backup = backup/
   ```

4. Put image files (`.jpg`) of your objects in the directory `data/obj/`

5. Create `train.txt` in directory `data/` with filenames of your images, each filename in new line, with path relative to `darknet`.

   For example containing:

   ```
   data/obj/img1.jpg
   data/obj/img2.jpg
   data/obj/img3.jpg
   ```

6. Download pre-trained weights for the convolutional layers and put to the directory `darknet` (root directory of the project)

   - for `yolov4.cfg`, `yolov4-custom.cfg` (162 MB): [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137)
   - for `yolov4-tiny.cfg`, `yolov4-tiny-3l.cfg`, `yolov4-tiny-custom.cfg`(19 MB): [yolov4-tiny.conv.29](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29)
   - for `csresnext50-panet-spp.cfg` (133 MB): [csresnext50-panet-spp.conv.112](https://drive.google.com/file/d/16yMYCLQTY_oDlCIZPfn_sab6KD3zgzGq/view?usp=sharing)
   - for `yolov3.cfg, yolov3-spp.cfg` (154 MB): [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
   - for `yolov3-tiny-prn.cfg , yolov3-tiny.cfg` (6 MB): [yolov3-tiny.conv.11](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)
   - for `enet-coco.cfg (EfficientNetB0-Yolov3)` (14 MB): [enetb0-coco.conv.132](https://drive.google.com/file/d/1uhh3D6RSn0ekgmsaTcl-ZW53WBaUDo6j/view?usp=sharing)

## Start training

```
./darknet detector train data/obj.data custom-yolov4-detector.cfg yolov4.conv.137 -dont_show
```

- file `yolo-obj_last.weights` will be saved to the `backup\` for each 100 iterations

- `-dont_show`: disable Loss-Window, if you train on computer without monitor (e.g remote server)

To see the mAP & loss0chart during training on remote server:

- use command `./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map`
- then open URL `http://ip-address:8090` in Chrome/Firefox browser)

After training is complete, you can get weights from `backu/`

### Notes

- If during training you see `nan` values for `avg` (loss) field - then training goes wrong! ​🤦‍♂️​

  But if `nan` is in some other lines - then training goes well.

- if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64

## Train tiny-YOLO

Do all the same steps as for the full yolo model as described above. With the exception of:

- Download file with the first 29-convolutional layers of yolov4-tiny:

   ```bash
  wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
   ```

  (Or get this file from yolov4-tiny.weights file by using command: `./darknet partial cfg/yolov4-tiny-custom.cfg yolov4-tiny.weights yolov4-tiny.conv.29 29`)

- Make your custom model `yolov4-tiny-obj.cfg` based on `cfg/yolov4-tiny-custom.cfg` instead of `yolov4.cfg`

- Start training: 

  ```bash
  ./darknet detector train data/obj.data yolov4-tiny-obj.cfg yolov4-tiny.conv.29
  ```

## Google Colab Notebook

[Colab Notebook](https://colab.research.google.com/drive/1aIc5xS8vVukVg-FiUA3aw0PUqYrXs8aO?authuser=1#scrollTo=Zz8v67_2kgWh)

### Small hacks to keep colab notebook training

1. Open up the inspector view on Chrome

2. Switch to the console window

3. Paste the following code

   ```javascript
   function ClickConnect(){
   console.log("Working"); 
   document
     .querySelector('#top-toolbar > colab-connect-button')
     .shadowRoot.querySelector('#connect')
     .click() 
   }
   setInterval(ClickConnect,60000)
   ```

   and hit **Enter**.

 It will click the screen every 10 minutes so that you don't get kicked off for being idle!

## Reference

- Guide from [AlexeyAB](https://github.com/AlexeyAB)/**[darknet](https://github.com/AlexeyAB/darknet)** repo: [How to train (to detect your custom objects)](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

- Tutorials

  - 👨‍🏫 How to Train YOLOv4 on a Custom Dataset in Darknet

    - [Colab Notebook](https://colab.research.google.com/drive/1mzL6WyY9BRx4xX476eQdhKDnd_eixBlG?authuser=0#scrollTo=QyMBDkaL-Aep)

    - Blog post: https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/

    - Video tutorial:

      {{< youtube N-GS8cmDPog >}}

    - [YOLOv4 - Ten Tactics to Build a Better Model](https://blog.roboflow.com/yolov4-tactics/)

  - YOLOv4 in the CLOUD: Build and Train Custom Object Detector (FREE GPU)

    - [Colab Notebook](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg#scrollTo=O2w9w1Ye_nk1)

    - Video tutorial:

      {{< youtube mmj3nxGT2YQ >}}

  - [Custom YOLOv4 Model on Google Colab](https://jkjung-avt.github.io/colab-yolov4/)

    - [Colab Notebook](https://colab.research.google.com/drive/1eoa2_v6wVlcJiDBh3Tb_umhm7a09lpIE?usp=sharing#scrollTo=J1oTF_YRoGSZ)

  - [TensorRT YOLOv4](https://jkjung-avt.github.io/tensorrt-yolov4/)

  - [YOLOv4 on Jetson Nano](https://jkjung-avt.github.io/yolov4/) 



