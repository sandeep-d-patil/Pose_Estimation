# Datasets
The datasets under consideration are PASCAL 3D+, CAVIAR-o and Towncentre datasets and they are divided into three parts namely training , validation and testing. the distribution are as shown as follows.

Dataset | PASCAL 3D+ | CAVIAR-o | Towncentre
------------ | -------------|-----------|---------|
Train set | 2247 | 10802 | 6916
Validation set| 562 |5444 | 874
Test set  | 275 |5445 | 904

[PASCAL 3D+](https://drive.google.com/file/d/1baI_QUNuGN9DJGgaOWubbservgQ6Oc4x/view) contains the 10 different classes of images with the size of 224x224x3. The truth values contain the three canonical angles. The airplane class data has been used to train, validate and test the model. Since for a deep learning network needs a large amount of dataset to learn the features, a large amount of images from the dataset have been used in the training. The validation set has been used in the model to influence the 'kappa' value. kappa value is a measure of concertration of the data around the mean value of the distribution. This plays a major role in increasing the probability of finding the accurate value of the object pose.

[CAVIAR-o](https://omnomnom.vision.rwth-aachen.de/data/BiternionNets/) dataset contains images of partially occluded heads , the images have been upscaled to 50x50x3 images from their original size of 7x7x3 images. the truth values contain the gaze angle in degrees. Due to availability of the more images , the number of validation set and testing set are increased. This dataset pose a challenge for the network due to two things mainly, upscale and blur in the image.

[Towncentre](https://omnomnom.vision.rwth-aachen.de/data/BiternionNets/) dataset contains images from the videoclip recorded from a surveillance camera. The images are of size 50x50x3. The truth values contain the gaze angle in degrees.  This dataset contains heads of tracked pedestrians in a shopping district, annotated with head pose regression labels.
