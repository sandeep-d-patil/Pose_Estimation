## Pose Estimation using VGG-style network and InceptionResNet

## Introduction
This blog aims to describe our efforts into reproducing the paper “Deep Directional Statistics: Pose Estimation with Uncertainty Quantification”. The paper discusses a method to perform tasks object pose estimation using uncertainty quantification. This uncertainty quantification allows for an increased robustness against images of varying quality.

The proposed method for uncertainty quantification consists of using a VGG-style convolutional network combined with a probabilistic von Mises distribution to predict the distribution over the object pose angle. The paper discusses three different types of von Mises distributions. First, where strong assumptions are made about the distribution. Second, where a finite number of mixture components determines the shape and third, where an infinite number of mixture components defines the shape. For this blog only the first two will be elaborated. The proposed method is tested on the PASCAL 3D+, CAVIAR-o and TownCentre datasets. 

The paper was provided with code, which was written in Tensorflow using the Keras high-level API. These software packages go about in a different way of building neural networks compared to Pytorch. The paper itself describes little about the setup of the code, and given that it is quite a complicated topic made rebuilding the code in Pytorch a difficult process. However, it did provide a good basis to learn on.

## Network Architecture
The architecture of the network is similar between the single density and finite mixture models. The network can be considered very deep and sequential with 24 layers. There are 5 convolution layers used which have 3x3 kernel sizes throughout. The volume reduction is taken care by the max pooling layer of size 2x2 which is used twice in the network. The Batch normalizations are used to normalize the values of the running averages which are 6 in number . ReLU (Rectified Linear Unit  is used as the activation functions . The Layers are flattened in the end using Linear layer and then Dropout is carried out in order to obtain more accurate weights of the network. The Network when used for training a 224x224x3 image, it has 78,781,822 trainable parameters. The total parameter size is about 300 MB. The network can be visualized as shown below.

The input for the network is a RGB image and the output is 1x6 tensor having predicted angles of the pose of the object in the image, which are in the bit format. This is not a conventional output format and has effects on usage of the standard loss functions such as cross entropy loss function.















Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/sandeeprockstar/Pose_Estimation/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact
This document is created by Sandeep Patil

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
