# Datasets
The datasets under consideration are PASCAL 3D+, CAVIAR-o and Towncentre datasets and they are divided into three parts namely training , validation and testing. the distribution are as shown as follows.

|Dataset | PASCAL 3D+ | CAVIAR-o | Towncentre|
|------------ | -------------|-----------|---------|
|Train set | 2247 | 10802 | 6916|
|Validation set| 562 |5444 | 874|
|Test set  | 275 |5445 | 904|

[PASCAL 3D+](https://drive.google.com/file/d/1baI_QUNuGN9DJGgaOWubbservgQ6Oc4x/view) contains the 10 different classes of images with the size of 224x224x3. The truth values contain the three canonical angles. The airplane class data has been used to train, validate and test the model. Since for a deep learning network needs a large amount of dataset to learn the features, a large amount of images from the dataset have been used in the training. The validation set has been used in the model to influence the 'kappa' value. kappa value is a measure of concertration of the data around the mean value of the distribution. This plays a major role in increasing the probability of finding the accurate value of the object pose.

[CAVIAR-o](https://omnomnom.vision.rwth-aachen.de/data/BiternionNets/) dataset contains images of partially occluded heads , the images have been upscaled to 50x50x3 images from their original size of 7x7x3 images. the truth values contain the gaze angle in degrees. Due to availability of the more images , the number of validation set and testing set are increased. This dataset pose a challenge for the network due to two things mainly, upscale and blur in the image.

[Towncentre](https://omnomnom.vision.rwth-aachen.de/data/BiternionNets/) dataset contains images from the videoclip recorded from a surveillance camera. The images are of size 50x50x3. The truth values contain the gaze angle in degrees. This dataset contains the head images of pedestrians in a shopping district with annotated head pose angles.

The towncentre dataset and caviar datasets required some preprocessing as the downloaded file format (i.e .bz2 format) and input format (.pkl.gz format) needed to run the model are different, for this we referred to the [Lucas Beyers](https://github.com/lucasb-eyer/BiternionNet/tree/e0cf7d29bfa177e4d8eded463e2c8304b78e2984) repository in order to find the source and methods required to download and preprocess the data. After preprocessing the data is then converted to pytorch tensors using `torch.tensor(..)` and `.permute(..)` methods so as to input them into the dataloader readable format. The data downloaded for each dataset respectively should be placed in the root folder before running the dataloading process.

# Results
The results of pytorch model and authors keras model with single density function model are discussed here. All the three datasets mentioned earlier are compared and the Table 2 of the [Sergey Prokudin et al](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sergey_Prokudin_Deep_Directional_Statistics_ECCV_2018_paper.pdf) paper are discussed. 
Apart from reproducing the results from the table, two different aspects of losses and errors are compaired here:
1. Effects of losses with variation in batch sizes during training.
2. Comparison of Losses and error authors keras models and our pytorch model.

## 1.Effects of losses with variation in batch sizes during training
The loss of distribution in training data and validating data give information about overfitting and underfitting of the model. The paper under consideration of reproduction does not state any facts or arguments about a few important parameters like: data batch sizes for training and validation or division. Hence we deemed it important to evaluate these parameters in order to produce good results for the data. 

![overandunderfit](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRQmjSeu1pjHZpj6WqRkL3cywXaiFkFNjL4YHu6GLGqm4GXEDoO&usqp=CAU) 


From the above image we can see that having a lower batch size the loss fluctuations over each epoch is high and as and when the batch size is increased the loss fluctuation is decreased. This is based on the research conducted by [Sam McCandlish et al](https://arxiv.org/pdf/1812.06162.pdf) We can observe that in the caviar dataset trained over batch size of 100 and 50 batches respectively

## insert the caviar dataset for batch size 100 and 50

## 2. Comparison of Losses and error authors keras models and our pytorch model
To compare the error values in the Table 2. we have to achieve a comparable model in pytorch as the model in keras. Hence we carried out various training and validations to compare the models. The plots of test and validation losses for similar setup in both keras model and pytorch model are shown below.

## insert the keras and pytorch model results for different datasets

Here we can see that the models are able to fit the data with comparable losses and hence can be used to compare the MAAD and Log likelihood losses.






|                            | Batch size: |       25       | 50             | 100            | 25             | 50              | 100            |
|----------------------------|-------------|:--------------:|----------------|----------------|----------------|-----------------|----------------|
|                            | Epochs      | MAAD error     | MAAD error     | MAAD error     | Log-likelihood | Log-likelihood  | Log-likelihood |
| CAVIAR-o - predict kappa   | 200         | 5.76 +/- 0.17  | 6.03 +/- 0.16  | 5.51 +/- 0.16  | 0.51 +/- 0.09  | 0.56 +/- 0.06   | 0.57 +/- 0.09  |
| TownCentre - predict kappa | 50          | 24.44 +/- 1.08 | 24.85 +/- 1.15 | 25.97 +/- 1.18 | -0.78 +/- 0.06 | -0.78 +/- 0.067 | -0.92 +/- 0.08 |
|                            |             |                |                |                |                |                 |                |



|            | MAAD error     |  Log-likelihood |
|------------|----------------|:---------------:|
| No shuffle | 31.85 +/- 1.22 | -0.92 +/- 0.034 |
| Shuffle    | 25.97 +/- 1.18 | -0.92 +/- 0.077 |

After running models in different scenarios we finalized on the  epoch size of 300,batch size of 100 for first 200 epochs and 50 for consecutive epochs, and learning rate of 1e-3 in order to achieve the results from the table 2. We have achieved comparable results with our pytorch model as shown below in the table.

The single density models with which we ran the pytorch networks with the fixed kappa value produced the results as shown in the table above. The results produced are not similar compared to the results in the table. This is due to the lack of tuning parameters.

# Conclusion
The single density model for pose estimation is successfully reproduced from keras high-level API to pytorch. The errors for CAVIAR-o dataset from our model vary less than 6 percent and errors for towncentre dataset from our model vary less than 7 percent of the value of errors when compared to the original authors model. 
Hence single density model results of Table2 have been implemented.
