# About this repository
This is a fork of Darknet, which can be found at the [Darknet project website](http://pjreddie.com/darknet).

# Using for detection
## Setup
1. Install [OpenCV
   2.4](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html).
2. (Recommended) Install CUDA 7.5 or above on NVIDIA GPU machine.
3. Clone the repository, checkout `racket` branch:
  ```
   git clone https://github.com/anujkhare/darknet-tennis
   cd darknet-tennis
   git checkout racket
  ```

4. Modify the Makefile, in particular, set the GPU, OpenCV flags, and the location of the CUDA
   installation:
  ```
   GPU=1
   OPENCV=1
   COMMON+= -DGPU -I/<path-to-cuda>/include/
   LDFLAGS+= -L/<path-to-cuda>/lib64 -lcuda -lcudart -lcublas -lcurand
  ```

5. Build repository:
  ```
   make -j8
  ```

6. Download the pretrained model (256 MB) in this folder.
  ```
   wget http://pjreddie.com/media/files/yolo.weights
  ```

You're all set!

## Usage
1. Run with single image:
  ```
   ./darknet detector test cfg/racket.data cfg/yolo-racket.cfg yolo.weights /path/to/image
  ```

2. Multiple images:
  ```
   ./darknet detector test cfg/racket.data cfg/yolo-racket.cfg yolo.weights
  ```

## Dockerfile (TODO)
Alternatively, you can use this Dockerfile to build it more easily. However:
- This might download ~1GB of data.


# Problem statement
Train simple model (CNN or other) to recognize image of a tennis racquet.
1. I would like you to Collect few images of tennis racquet from web. Use them to train a simple model, preferably a CNN.
2. Would like to test model with a tennis racquet image from our camera.
3. Please spell out the assumptions you would make on training images. We will follow same assumption on test image.
4. Please give us training and recognition code, as two different executables. So that we understand deployment of trained models.

# Analysis
## One-class SVM
We could train one class SVM with only training images of tennis rackets, to
learn the distribution of training rackets. For a new image, the SVM would
accept images similar to the training images, and reject others.

## 2-class classification using CNN
This problem can be formulated as a 2-class discriminatory problem (tennis-racket
vs everything-else), using a CNN.

However, it is not well defined what the images the other class should contain.

## Object detection using CNN
A related problem is to detect a tennis racket in an image by drawing a
bounding box around the object.

There are several popular models available that perform object detection using
CNNs.

*I explored two different object detection models for this problem:*
- Faster-RCNN: This proved to be very hard to get to build with all the
  dependencies. I uploaded a Dockerfile for building it [here](). Overall,
  given it's complexity to work with, I chose to look for alternatives.

- You-Only-Look-Once ([YOLO](http://pjreddie.com/darknet/yolo)): This is a real-time object detection pipeline
  built using C and CUDA.

## Training YOLO for racket images
*I will upload instructions to setup the training data and train the model
later.*

I took a YOLO model pre-trained on MS-COCO (which already has a tennis-racket
class), and just suppressed the output for all other classes. As a result, the
network only outputs the bounding boxes for tennis rackets.

A more proper approach is to train a network starting from an extraction model
(based on imagenet).

**At the time of writing, the model is being trained on the ~1500 images of
rackets for the past 1 day on my computer, and would probably take a few more
days to complete.**


## Data
For object detection, we need images + labels of the bounding boxes. From the
well-known datasets, I found tennis rackets in:
- [ImageNet](imagenet.stanford.edu/synset?wnid=n04039381): racket synset containing 458 labelled images of rackets
  (including badminton, squash, and other rackets).

- [MS COCO](mscoco.org/dataset/#download): tennis-racket class containing over 1000 annotated images.
  Downloaded using
  [this](https://gist.github.com/anujkhare/91413d1c6524bd917d37ece541578b5e) script.

Small scripts
[1](https://gist.github.com/anujkhare/28738577df405b29d211b88594357173) were used to convert the annotations into the desired
format for this model.

