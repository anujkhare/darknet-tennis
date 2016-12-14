# About this repository
This is a fork of Darknet, which can be found at the [Darknet project website](http://pjreddie.com/darknet).

# Using for detection
## Setup
1. Install [OpenCV
   2.4](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html).
2. (Optional) Install CUDA 7.5 or above on NVIDIA GPU machine.
3. Clone the repository, checkout `racket` branch:
  ```
  git clone https://github.com/anujkhare/darknet-tennis
  cd darknet-tennis
  git checkout racket
  ```
4. Download the pretrained model.
5. 



## Dockerfile
Alternatively, you can use this Dockerfile to build it more easily. However:
- Real-time detection *won't work*.
- This might download ~1GB of data.

1. Install docker and nvidia-docker


# Problem
Train simple model (CNN or other) to recognize image of a tennis racquet.
1. I would like you to Collect few images of tennis racquet from web. Use them to train a simple model, preferably a CNN.
2. Would like to test model with a tennis racquet image from our camera.
3. Please spell out the assumptions you would make on training images. We will follow same assumption on test image.
4. Please give us training and recognition code, as two different executables. So that we understand deployment of trained models.

# Analysis
## One-class SVM

## 2-class classification using CNN
This problem can be formulated as a 2-class discrimatory problem (tennis-racket
vs everything-else). A CNN can be trained on these images to perform the
classification.

However, it is not well defined what the images the other class should contain, 


## Object detection using CNN
A related problem is to detect a tennis racket in an image by drawing a
bounding box around the object.

There are several popular models available that perform object detection using
CNNs.

*I explored two different object detection models for this problem:*
- Faster-RCNN: This proved to be very hard to get to build with all the
  dependencies. I uploaded a Dockerfile for building it [here](). Overall,
  given it's complexity to work with, I chose to look for alternatives.

- You-Only-Look-Once (YOLO) (Darknet): This is a real-time object detection pipeline
  built using C and CUDA.

## Training YOLO for racket images
I will upload instructions to setup the training data and train the model
later.

I took a YOLO model pre-trained on MS-COCO (which already has a tennis-racket
class), and just suppressed the output for all other classes. As a result, the
network only outputs the bounding boxes for tennis rackets.

A more proper approach is to train a network starting from an extraction model
(based on imagenet).

*At the time of writing, the model is being trained on the ~1500 images of
rackets for the past 1 day on my computer, and would probably take around 5
days to complete.*


## Data
For object detection, we need images + labels of the bounding boxes. From the
well-known datasets, I found tennis rackets in:
- [ImageNet](): racket synset containing 458 labelled images of rackets
  (including badminton, squash, and other rackets).

- [MS COCO](): tennis-racket class containing over 1000 annotated images.
  Downloaded using [this]() script.

Small scripts [1]() were used to convert the annotations into the desired
format for this model.

