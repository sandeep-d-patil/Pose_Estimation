## Pose Estimation using VGG-style network and InceptionResNet

## Introduction
This blog aims to describe our efforts into reproducing the paper “Deep Directional Statistics: Pose Estimation with Uncertainty Quantification”. The paper discusses a method to perform tasks object pose estimation using uncertainty quantification. This uncertainty quantification allows for an increased robustness against images of varying quality.

The proposed method for uncertainty quantification consists of using a VGG-style convolutional network combined with a probabilistic von Mises distribution to predict the distribution over the object pose angle. The paper discusses three different types of von Mises distributions. First, where strong assumptions are made about the distribution. Second, where a finite number of mixture components determines the shape and third, where an infinite number of mixture components defines the shape. For this blog only the first two will be elaborated. The proposed method is tested on the PASCAL 3D+, CAVIAR-o and TownCentre datasets. 

The paper was provided with code, which was written in Tensorflow using the Keras high-level API. These software packages go about in a different way of building neural networks compared to Pytorch. The paper itself describes little about the setup of the code, and given that it is quite a complicated topic made rebuilding the code in Pytorch a difficult process. However, it did provide a good basis to learn on.

## Network Architecture
The architecture of the network is similar between the single density and finite mixture models. The network can be considered very deep and sequential with 24 layers. There are 5 convolution layers used which have 3x3 kernel sizes throughout. The volume reduction is taken care by the max pooling layer of size 2x2 which is used twice in the network. The Batch normalizations are used to normalize the values of the running averages which are 6 in number . ReLU (Rectified Linear Unit  is used as the activation functions . The Layers are flattened in the end using Linear layer and then Dropout is carried out in order to obtain more accurate weights of the network. The Network when used for training a 224x224x3 image, it has 78,781,822 trainable parameters. The total parameter size is about 300 MB. The network can be visualized as shown below.

The input for the network is a RGB image and the output is 1x6 tensor having predicted angles of the pose of the object in the image, which are in the bit format. This is not a conventional output format and has effects on usage of the standard loss functions such as cross entropy loss function.

## Dataset and DataLoader
All datasets used in the paper are not standard sets that are included within the Pytorch computer vision package, torchvision. Therefore, we have to write our own dataloader class that will be used later on to run the batches in the training process. 

First, we load the PASCAL3D+ dataset using a script provided by the author to split the dataset in a training, validation and test set. Second, we inherit the functionality of DataSet from torch.utils.data in our dataloader, which is done by overwriting the '__len__' and '__getitem__' methods. Next, a train_set is d

```from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import pascal3d

'<cls = 'aeroplane' # if cls is None, all classes will be loaded>'
pascaldb_path = 'data/pascal3d+_imagenet_train_test.h5'
x_train_tf, y_train_tf, x_val_tf, y_val_tf, x_test_tf, y_test_tf = pascal3d.load_pascal_data(pascaldb_path, cls=cls)

# preparing the pascal3d dataset for the pytorch environment
x_train = (torch.tensor(x_train_tf[:])).permute(0, 3, 1, 2).float() 
y_train = torch.tensor(y_train_tf[:])
x_val = (torch.tensor(x_val_tf[:])).permute(0, 3, 1, 2).float() 
y_val = torch.tensor(y_val_tf[:])
x_test = (torch.tensor(x_test_tf[:])).permute(0, 3, 1, 2).float() 
y_test = torch.tensor(y_test_tf[:])

class dataloader(Dataset):
  def __init__(self, samples, labels):
    self.labels = labels
    self.samples = samples

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, index):
    sample = self.samples[index]
    label = self.labels[index]
    return sample, label

train_set = dataloader(x_train, y_train)
val_set = dataloader(x_val, y_val)
test_set = dataloader(x_test, y_test)

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {} 

train_loader = DataLoader(train_set, batch_size=5, shuffle=False, **kwargs)
val_loader = DataLoader(val_set, batch_size=5, shuffle=False, **kwargs)
test_loader = DataLoader(test_set, batch_size=5, shuffle=False, **kwargs)
```













Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for


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
