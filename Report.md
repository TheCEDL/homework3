# Homework3 - Generative Models
Member: 簡廷安, 巫姿瑩, 胡展維
## Breif
The DCGAN model in this [paper](https://arxiv.org/pdf/1511.06434v2.pdf) can only generate low resolution images(64x64). So we use the structure from a recent
[paper](https://arxiv.org/pdf/1612.03242v1.pdf) called StackGAN, and it can generate high resolution images(256x256. The model is devided into two stages and trained separately.
The Stage-I GAN sketches the primitive shape and basic colors of the object, yielding Stage-I low resolution images. The
Stage-II GAN takes Stage-I results as inputs, and generates high resolution images with photorealistic details. The Stage-II GAN is able to rectify defects and add compelling details with the refinement process.
Need to mention that in this paper it generates images conditioned on text descriptions while we didn't do this part.
 
![](https://github.com/gina9726/homework3/blob/master/images/model.png)
## Stage-I
The structure in stage-I is similar to the DCGAN structure. We modify the DCGAN-tensorflow code to complete this part.
  
## Stage-II
In stage-II we firstly downsampled the images which are generated from stage-I, then fed into 4 residual blocks(consist of 3 × 3 stride 1 convolutions, Batch normalization and ReLU) described in the paper. The residual block is to encode the image, and finally a series of up-sampling blocks are used to generate a 256 x 256 image.
For the discriminator, its structure is similar to that of Stage-I discriminator with only extra down-sampling blocks since the image size is larger in this stage.

## Training details
* Stage-I
    * Learning rate starts with 0.0002, and is decayed exponentially.
    * Input real images are resized to 64x64, and scaled to be bounded in [-1, 1].
    * We train 40 epochs to get stable generated images.
* Stage-II
    * Learning rate starts with 0.0002, and is decayed exponentially.
    * Input real images are 256x256, and scaled to be bounded in [-1, 1].
    * The fake images(64x64) generated from stage-I are downsampled to 32x32, and upsampled to 256x256 in stage-II. 
    * When training stage-II, the weights of stage-I model are fixed. We only forward the generator in stage-I to get the fake images (64x64).
    * We train 26 epochs to get the result.

## Result
* Stage-I (64x64)  

![](https://github.com/gina9726/homework3/blob/master/images/stageI.jpg) 

* Stage-II (256x256)  

![](https://github.com/gina9726/homework3/blob/master/images/stageII.jpg)

## Reference paper
[UNSUPERVISED REPRESENTATION LEARNING
WITH DEEP CONVOLUTIONAL
GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434v2.pdf)

[StackGAN: Text to Photo-realistic Image Synthesis
with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242v1.pdf)
