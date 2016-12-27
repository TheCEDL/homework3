## Homework3 - Generative Models

## Member
* captain - <a href="https://github.com/maolin23?tab=repositories">林姿伶</a>: 104062546
* member -  <a href="https://github.com/hedywang73?tab=repositories">汪叔慧</a>: 104062526
* member - <a href="https://github.com/yenmincheng0708?tab=repositories">嚴敏誠</a>: 104062595

` Contribution `
```
姿伶 and 叔慧 discussed the homework.
Finally, 姿伶 typed the final report on github.
```

## Introduction

　　　　　![Fig. 1](https://github.com/CEDL739/homework3/blob/master/img/Value_Func_of_GAN.JPG)<br>
　　　　　　　　　　　　　　　　　　　　**Fig .1** Value Function of GAN<br>

　　This is the objective function of GAN, D() is a discriminator and G() is a generator.<br>
　　The goal of the generator is to generate a fake image that looks like a real image. And the goal of the discriminator is to judge the fake and the real image.<br><br>
　　Therefore, it’s a competitive relationship between the discriminator and generator.<br><br>
　　In this homework we try some settings. The first one is directly modify the output's size to 256 x 256(model 1). However, with this setting the feature map of the first layer would be 16 x16. In order to remain the size of the first layer’s feature map, we makes the network deeper(model 2,4). After adding two convolution layers, feature maps of this deeper network would be the size of [4 8 16 32 64 128 256]. Finally, since the kernel size of this network is 5x5, we also modify this size to 3x3 to test whether the smaller kernel size could improve the result(model 3,4).<br><br>
   
Model | Number of Layers | Kernel size
------|------------------|------------
　1 |　　　　4　　|　　5*5
　2 |　　　　6　　|　　5*5
　3 |　　　　4　　|　　3*3
　4 |　　　　6　　|　　3*3


## Experiment setting

### Generator

* Deeper is better?<br>
  - From the experiment result, we observe that the deeper network could generate a good fake image, but it tends to generate the same result.(model 2,4)<br>
* Kernel size<br>
  - Kernel size with 3x3, compared to 5x5, is much more likely to generate some strange patterns.<br>
* Through the observation of experiment results, we finally choose setting 1(directly modify output's size to 256x256) as the experiment setting. Though the deeper network could generate a good image, it couldn’t generate a large number of different images. For the sake of generation, we finally choose setting 1 as our final result.<br>

### Discriminator

Through the evaluation result, we observe that the discriminator we trained tends to judge the image as the fake image.<br>


## Results

![Fig. 2](https://github.com/CEDL739/homework3/blob/master/img/test_model1.png)<br>
　　　　　　　　　　　　　　　　　　　　　　　**Fig. 2** model 1<br>

![Fig. 3](https://github.com/CEDL739/homework3/blob/master/img/test_model2.png)<br>
　　　　　　　　　　　　　　　　　　　　　　　**Fig. 3** model 2<br>

![Fig. 4](https://github.com/CEDL739/homework3/blob/master/img/test_model3.png)<br>
　　　　　　　　　　　　　　　　　　　　　　　**Fig. 4** model 3<br>

![Fig. 5](https://github.com/CEDL739/homework3/blob/master/img/test_model4.png)<br>
　　　　　　　　　　　　　　　　　　　　　　　**Fig. 5** model 4<br>
                       
![Fig. 6](https://github.com/CEDL739/homework3/blob/master/img/good_result.png)<br>
　　　　　　　　　　　　　　　　　　　 **Fig. 6** selected good results from model 1<br>                    
