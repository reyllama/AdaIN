# Pytorch-AdaIN



Based on [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf)

## Architecture

![image-20210109145807459](/assets/images/architecture.JPG)

- Adaptive Instance Normalization

  ![image-20210109145842222](/assets/images/adain.JPG)

Basic Idea is to align the pixelwise statistics between content and style image feature maps. Use pretrained VGG-19 for VGG Encoder and mostly mirrored architecture for Decoder, mainly with batch normalization layers removed. The details are stated in the paper.



- Content and Style Loss

  ![image-20210109150047135](/assets/images/loss1.JPG)

  - Content Loss

    ![image-20210109150106985](/assets/images/loss2.JPG)

    - Euclidean distance between bottleneck feature maps. This refers to the reconstruction loss.

  - Style Loss

    ![image-20210109150225621](/assets/images/loss3.JPG)

    - Euclidean distance between feature map statistics from relu_1_1, relu_2_1, relu_3_1, relu_4_1. This assures that two moments of transformed image and original style images be close.

- (Note) Original paper is officially implemented in lua.



## Photo-Monet Translation Output

- Sample content images

![Image](/assets/images/content.JPG)

- Sample Style Images

![image](/assets/images/style.JPG)

- AdaIN Output

  - Interpolating between content-style with value $0 \geq \alpha \leq 1$

  - First column: style, second column: content, others: transformed with $\alpha$ = 0.2, ..., 1.0

  ![image](/assets/images/out1.JPG)

  ![image](/assets/images/out2.JPG)

  ![image](/assets/images/out3.JPG)

  ![image](/assets/images/out4.JPG)

  ![image](/assets/images/out5.JPG)