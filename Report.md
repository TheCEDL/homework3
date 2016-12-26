#CEDL Homework 3 Report

team member: 林沅廷, 石孟立, 王福恩, 王尊玄

##Introduction:
In this homework, we will practice to use state-of-the-art generative model (GAN) to produce higher resolution image(compared to traditional 64*64 image) In the following session, we will elucidate how we modify provided DCGAN to ahieve the target resolution.

##Implementation:
To make DCGAN capable to generate higher resolution, the simplest way is using the same architecture as DCGAN but with 2 more deconvolution layers (picture below is the 64*64 version DCGAN generator)
<center><img src="./pic/dcgan.png" width=99%></center>

However, the training process in 64*64 DCGAN is already hard enough. It's quite unrealistic to add two more deconvolution layers directly since the model might be harder to train and unstable.

In recent works, there are many modyfied GAN architecture such as infoGAN, pixel2pixel and stackGAN. some for conditional control and some for higher resolution. In this project, we are inspired by the concept of DCGAN. we propose a two stage cascaded generator architecture
##Generated data:

google drive link= https://drive.google.com/open?id=0B3RbJqaypz2TWDJ1U3dtNzhfVlE


##Discussion:
##Reference: