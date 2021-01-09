# Pytorch-AdaIN



Based on [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf)

## Architecture

![image-20210109145807459](/asset/images/architecture.jpg)

- Adaptive Instance Normalization

  ![image-20210109145842222](/asset/images/adain.jpg)

Basic Idea is to align the pixelwise statistics between content and style image feature maps. Use pretrained VGG-19 for VGG Encoder and mostly mirrored architecture for Decoder, mainly with batch normalization layers removed. The details are stated in the paper.



- Content and Style Loss

  ![image-20210109150047135](/asset/images/loss1.jpg)

  - Content Loss

    ![image-20210109150106985](/asset/images/loss2.jpg)

    - Euclidean distance between bottleneck feature maps. This refers to the reconstruction loss.

  - Style Loss

    ![image-20210109150225621](/asset/images/loss3.jpg)

    - Euclidean distance between feature map statistics from relu_1_1, relu_2_1, relu_3_1, relu_4_1. This assures that two moments of transformed image and original style images be close.

- (Note) Original paper is officially implemented in lua.



## Photo-Monet Translation Output

