## Pose Estimation using VGG-style network and InceptionResNet
The link for the paper which is reproduced is [here](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sergey_Prokudin_Deep_Directional_Statistics_ECCV_2018_paper.pdf).
## Introduction
This blog aims to describe our efforts into reproducing the paper “Deep Directional Statistics: Pose Estimation with Uncertainty Quantification”. The paper discusses a method to perform tasks object pose estimation using uncertainty quantification. This uncertainty quantification allows for an increased robustness against images of varying quality.

The proposed method for uncertainty quantification consists of using a VGG-style convolutional network combined with a probabilistic von Mises distribution to predict the distribution over the object pose angle. The paper discusses three different types of von Mises distributions. First, where strong assumptions are made about the distribution. Second, where a finite number of mixture components determines the shape and third, where an infinite number of mixture components defines the shape. For this blog only the first two will be elaborated. The Pascal 3D+ dataset is used as testbed to study the general object pose estimation. The CAVIAR-o and TownCentre datasets presents a challenging task of corase gaze estimation as the images are of low resolution which are obtained from surveillance camera videos. In the CAVIAR dataset, the occulded the head instances are considered for testing. Hence this paper aims to produce a Deep neural network focussed on head pose detection in crowded places from the surveillance cameras. 

The paper was provided with code, which was written in Tensorflow using the Keras high-level API. These software packages go about in a different way of building neural networks compared to Pytorch. The paper itself describes little about steps followed to achieve the desired results, and given that it is quite a complicated topic made rebuilding the code, from TensorFlow, in Pytorch a difficult process. However, it did provide a good basis to learn on.

# Understanding the original code
As we both are novices in both Pytorch and Tensorflow understanding the original code was already quite a big task. Most of the code was uncommented and we were not able to run the code out of the box for any of the datasets/loss-function scenarios. In order to fully understand the deep neural net proposed in the paper the main focus was to get the single density model running for the PASCAL3D+ dataset. This was considered an essential addition to the explanation in the paper to understand what was happening. 

We started out by learning how a neural network is built and trained within Tensorflow. This meant getting to grips with the functional Keras API that is used. The propagation of information throughout the model is dependent on which model you run. The model options are elaborated below.
 
 ADD IMAGE OF THE INFORMATION FLOW THROUGH THE MODEL!?
  
2. Maximizing von Mises log likelihood with a predicted kappa value.
     * Initialization
       * The model is initialized using `BiternionVGG(loss_type='vm_likelihood', predict_kappa=True)`.
       * Run `_pick_loss` method thereby setting loss equal to `von Mises log likelihood`, which is the negative von Mises Log Likelihood with corresponding inputs: ground truth in biternion angles, predicted biternion angles and predicted kappa value.
       * Define symbolic input X shape using `Input()`.
       * Feed symbolic input through the VGG backbone using `vgg_model(...)(input)`. Output is named vgg_x.
       * Feed symbolic output, `vgg_x`, through a fully connected layer with 2 outputs (angle prediction) and normalize output with L2 normalization. 
       * Feed symbolic output, `vgg_x`, ALSO through a fully connected layer with 1 output (kappa prediction) and make the output absolute. 
       * Since predict_kappa = True, the feedforward is defined using `Model(input_X, concatenate[y_pred, kappa_pred])`. This maps the symbolic input X through the above defined network to the final output. Note that this time the network splits up after the VGG backbone into two different dense layers followed by two different lambda layers. The outputs are then combined again forming 1 output. The feedforward is defined as names `model`.
       * Define the optimizer that will be used by running `keras.optimizers.Adam()`.
       * Compile the symbolic network using `model.compile()`. Note that model is our defined network. Further inputs are the loss function and optimizer. 
    * Training:
       * exactly the same as with model number 1
    * Validation:
       * validation is now solely for the purpose of checking the accuracy and loss after every epoch. Check tomorrow more securely!




# Setting up the Google Colab environment
The first step in the code building process is to setup the Google Colab environment. We do this by connecting Google Colab to Google Drive and setting the working directory to the right folde. All relevant documentation is uploaded to the `deep_direct_stat-master` folder which can be accessed directly from the Colab document. 

```markdown
import os
from google.colab import drive
drive.mount("/content/drive")
os.chdir('/content/drive/My Drive/Deep Learning/deep_direct_stat-master')
```

## Network Architecture
The architecture of the network is similar between the single density and finite mixture models. The network can be considered very deep and sequential with 24 layers. There are 5 convolution layers used which have 3x3 kernel sizes throughout. The volume reduction is taken care by the max pooling layer of size 2x2 which is used twice in the network. The Batch normalizations are used to normalize the values of the running averages which are 6 in number . ReLU (Rectified Linear Unit  is used as the activation functions . The Layers are flattened in the end using Linear layer and then Dropout is carried out in order to obtain more accurate weights of the network. The Network when used for training a 224x224x3 image, it has 78,781,822 trainable parameters. The total parameter size is about 300 MB. The network can be visualized as shown below.

The single mixture model can be visualized as below. There is only one simple von mises distribution used to obtain the pose of the object. Hence only one distribution can be seen as the output in the network.
![singlemixture](images/singlemixture.JPG)

The finite mixture model can be visualized as below. A complex distribution is generated by summing up multiple distribution in the case of finite mixture model. Here each component is a simple von mises distribution. Hence a finite number of distribution can be seen in the output of the network.
![Finitemixture](images/Finitemixture.png)

The input for the network is a RGB image of size 224x224 or 50x50 for respectively the PASCAL3D+ and Towncentre/CAVIAR-o datasets. The network has two outputs, containing the biternion azimuth/gaze angle, and additionally a third when kappa is predicted by the network. The VGG backbone of the network is provided below. Note the influence of `predict_kappa` in the forward function on the information propagation in the network.

```markdown
class vgg_model(nn.Module):
  def __init__(self, n_outputs=1, conv_dropout_val=0.2, 
               fc_dropout_val=0.5, fc_layer_size=512):
      super(vgg_model, self).__init__()
      self.VGG_backbone = nn.Sequential(
          nn.Conv2d(3, 24, kernel_size=3, stride=1), 
          nn.BatchNorm2d(24),         
          nn.ReLU(),               
          nn.Conv2d(24, 24, kernel_size=3, stride=1), 
          nn.BatchNorm2d(24), 
          nn.MaxPool2d(2),            
          nn.ReLU(),
          nn.Conv2d(24, 48, kernel_size=3, stride=1),
          nn.BatchNorm2d(48),
          nn.ReLU(),
          nn.Conv2d(48, 48, kernel_size=3, stride=1),
          nn.BatchNorm2d(48),
          nn.MaxPool2d(2),
          nn.ReLU(),
          nn.Conv2d(48, 64, kernel_size=3, stride=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Dropout2d(conv_dropout_val), 
          nn.Flatten(),
          nn.Linear(5*5*64, 512),      
          nn.ReLU(),                                                          
          nn.Dropout2d(fc_dropout_val))                                       
      self.final_layer = nn.Linear(512, n_outputs)
      self.kappa_predict = nn.Linear(512, 1)
      self.ypred = nn.Linear(512, 2)

  def forward(self, input, predict_kappa=True, final_layer=False, l2_normalize_final=False): 
      x_vgg = self.VGG_backbone(input)
      if final_layer:
          x = self.final_layer(x_vgg)
      if l2_normalize_final:
          x = F.normalize(x,dim=1,p=2)
      if not final_layer:
          if predict_kappa:
              x_ypred = F.normalize(self.ypred(x_vgg), dim=1,p=2)
              x_kappa = torch.abs(self.kappa_predict(x_vgg))
              x = torch.cat((x_ypred, x_kappa), 1)
          if not predict_kappa:
              x = self.ypred(x_vgg)
      return x
```
## Dataset and DataLoader
All datasets used in the paper are not standard sets that are included within the Pytorch computer vision package, torchvision. Therefore, we have to write our own DataSet class that will be used later on to run the batches in the training process. With this DataSet class we can access all the training samples in the dataset. The first step is to load the datasets, which is done with a script provided by the author that split the dataset in a training, validation and test set. Second, we inherit the functionality of the DataSet class in our dataloader, which is done by overwriting the `__len__` and `__getitem__` methods. The defined DataSet class now serves as input for the DataLoader class, which additionally accepts the parameter batch_size. The DataLoader is used to run through the data in the training process of our model.

```markdown
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import pascal3d
from datasets import caviar
from datasets import towncentre

# Loading and preparing the PASCAL3D+ dataset
cls = 'aeroplane' # if cls is None, all classes will be loaded
pascaldb_path = 'data/pascal3d+_imagenet_train_test.h5'
x_train_tf, y_train_tf, x_val_tf, y_val_tf, x_test_tf, y_test_tf = pascal3d.load_pascal_data(pascaldb_path, cls=cls)

x_train = (torch.tensor(x_train_tf[:])).permute(0, 3, 1, 2).float()
y_train = torch.tensor(y_train_tf[:])
x_val = (torch.tensor(x_val_tf[:])).permute(0, 3, 1, 2).float() 
y_val = torch.tensor(y_val_tf[:])
x_test = (torch.tensor(x_test_tf[:])).permute(0, 3, 1, 2).float() 
y_test = torch.tensor(y_test_tf[:])

# Loading and preparing the CAVIAR-o dataset
caviar_path = 'data/CAVIAR-o.pkl.gz'
(xtr_cav, ytr_cav_deg, info_tr), (xval_cav, yval_cav_deg, info_val), (xte_cav, yte_cav_deg, info_te) = caviar.load_caviar(caviar_path)

ytr_cav_bit = deg2bit(ytr_cav_deg)
yval_cav_bit = deg2bit(yval_cav_deg)
yte_cav_bit = deg2bit(yte_cav_deg)
xtr_cav = (torch.tensor(xtr_cav[:])).permute(0, 3, 1, 2).float() 
ytr_cav_bit = torch.tensor(ytr_cav_bit[:])
xval_cav = (torch.tensor(xval_cav[:])).permute(0, 3, 1, 2).float() 
yval_cav_bit = torch.tensor(yval_cav_bit[:])
xte_cav = (torch.tensor(xte_cav[:])).permute(0, 3, 1, 2).float() 
yte_cav_bit = torch.tensor(yte_cav_bit[:])

# Loading and preparing the TownCentre dataset
towncentre_path = 'data/TownCentre.pkl.gz'
(xtr_tc, ytr_tc_deg, img_names_tr), (xval_tc, yval_tc_deg, img_names_val), (xte_tc, yte_tc_deg, img_names_te) = towncentre.load_towncentre(towncentre_path)

ytr_tc_bit = deg2bit(ytr_tc_deg)
yval_tc_bit = deg2bit(yval_tc_deg)
yte_tc_bit = deg2bit(yte_tc_deg)
xtr_tc = (torch.tensor(xtr_tc[:])).permute(0, 3, 1, 2).float() 
ytr_tc_bit = torch.tensor(ytr_tc_bit[:])
xval_tc = (torch.tensor(xval_tc[:])).permute(0, 3, 1, 2).float() 
yval_tc_bit = torch.tensor(yval_tc_bit[:])
xte_tc = (torch.tensor(xte_tc[:])).permute(0, 3, 1, 2).float() 
yte_tc_bit = torch.tensor(yte_tc_bit[:])


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

# select dataset
load_dataset = 'towncentre'

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

if load_dataset == 'pascal':
    train_set = dataloader(x_train, y_train[:, 0:2])
    val_set = dataloader(x_val, y_val[:, 0:2])
    test_set = dataloader(x_test, y_test[:, 0:2])

    train_loader = DataLoader(train_set, batch_size=15, shuffle=False, **kwargs)
    val_loader = DataLoader(val_set, batch_size=15, **kwargs) 
    test_loader = DataLoader(test_set, batch_size=15, **kwargs)

elif load_dataset == 'caviar':
      train_set = dataloader(xtr_cav, ytr_cav_bit[:, 0:2])
      val_set = dataloader(xval_cav, yval_cav_bit[:, 0:2])
      test_set = dataloader(xte_cav, yte_cav_bit[:, 0:2])

      train_loader = DataLoader(train_set, batch_size=5, shuffle=False, **kwargs)
      val_loader = DataLoader(val_set, batch_size=15, **kwargs) 
      test_loader = DataLoader(test_set, batch_size=15, **kwargs)

elif load_dataset == 'towncentre':
      train_set = dataloader(xtr_tc, ytr_tc_bit[:, 0:2])
      val_set = dataloader(xval_tc, yval_tc_bit[:, 0:2])
      test_set = dataloader(xte_tc, yte_tc_bit[:, 0:2])

      train_loader = DataLoader(train_set, batch_size=5, shuffle=True, **kwargs)
      val_loader = DataLoader(val_set, batch_size=15, **kwargs) 
      test_loader = DataLoader(test_set, batch_size=15, **kwargs)     

data_loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader} 
```
The airplane class data has been used to train, validate and test the model. The test dataset has 2247 images. The validation and testing datasets have 562 and 275 images respectively. Since for a deep learning network needs a large amount of dataset to learn the features, a large amount of images from the dataset have been used in the training. The validation set has been used in the model to influence the 'kappa' value. kappa value is a measure of concertration of the data around the mean value of the distribution. This plays a major role in increasing the probability of finding the accurate value of the object pose.

The visualization of variation of kappa values for a distribution can be seen below. Higher the kappa value concentrates the data towards the centre of the distribution.

![kappa](/images/kappa.JPG)

The Pascal 3D+ datasets are visualized below. 
```markdown
import matplotlib.pyplot as plt

sample = next(iter(train_loader))
image, label = sample

grid = torchvision.utils.make_grid(image[0:10], nrow=10 )
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))
```
![loadimages](/images/loadimage.jpeg)

## Training, Validation and Evaluation
The training algorithm used in the Pytorch implemenation is illustrated below. This is much different to the Keras implementation, where 1 line of code suffices to start training a model. As previously explained, we iterate over the created dataloader to provide the training algorithm with the batches of images. All computations happen via the GPU. 
```markdown
def train(train_loader, model, max_epochs, optimizer, criterion):
  for epoch in range(max_epochs): 
    running_loss = 0.0
    for i, data in enumerate(train_loader):
      local_batch, local_labels = data
      # Transfer the batches to the GPU
      local_batch, local_labels = local_batch.to(device), local_labels.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + loss + backward + optimize
      outputs = model(local_batch)
      loss = cosine_loss_nn(local_labels, outputs)

      loss.backward(torch.ones_like(loss))
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 100 == 99:    
          print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
          running_loss = 0.0

  print('Finished Training')
  PATH = 'data/model_weights_single_density.pth'
  torch.save(model.state_dict(), PATH)
```

The code for the validation and evaluation are omitted as they still require some cleaning. As explained earlier, the validation set is used to tune parameter kappa and the evaluation set is to verify the fit of the model with regards to log likelihood and Mean Absolute Angular Deviation (MAAD).

## Results
In order to verify that the entire Pytorch model is working appropriatly the network is trained with the pascal 3D+ datasets as explained earlier. The obtained result are not yet satisfactoy, which is concluded from the high errors and strange values for kappa. This is most likely the result of a small amount of training time and additionally some required parameter/code tuning here and there. Hence the results are not displayed. We will carry out more training of the data in coming days and present sufficient results for the CAVIAR-o, TownCentre and PASCAL3D+ datasets.
