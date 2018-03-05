## StartGAN-Tensorflow. 
* Tensorflow implement of StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation.[[Paper]](https://arxiv.org/abs/1711.09020)
* Borrowed code and ideas from yunjey's StarGAN: https://github.com/yunjey/StarGAN.

## Install Required Packages
First ensure that you have installed the following required packages:
* TensorFlow1.4.0 ([instructions](https://www.tensorflow.org/install/)). Maybe other version is ok.
* Opencv ([instructions](https://github.com/opencv/opencv)). Here is opencv-2.4.9.

See requirements.txt for details.

## Datasets
* In this implementation of the StarGAN, we use CelebA dataset.
* Run the **download.sh** to download the dataset, you can put it in the datasets folder. The CelebA dataset which you downloaded inculdes CelebA crop images and attribute labels.
* CelebA attribute labels can be seen in list_attr_celeba.txt, there are 40 attributes totally. It is a binary attribute, 1 represent the image has this attribute, -1 represent the image has not this attribute. You can see the README.md of CelebA dataset for detial.

## Training and Testing Model
* Run the following script to train the model, in the process of training, will save the training images every 500 steps. See the **model/stargan.py** for details.
```shell
sh train.sh
```
You can change the arguments in train.sh depend on your machine config.
* Run the following script to test the trained model. The test.sh will transform the datasets.
```shell
sh test.sh
```
The script will load the trained StarGAN model to generate the transformed images. You could change the arguments in test.sh depend on your machine config.

## Downloading data/trained model
* Pretrained model: [[download]](https://drive.google.com/open?id=1ngSzJN3oUdn2Xrrvl_vNyPsQThI0hHcY).

## Discussion
* In the process of training, you will see the loss change greatly, and the generated results is also not good. I guess the loss function is not well, so, we can try other loss function. If you have other ideas about my project, please contact me.