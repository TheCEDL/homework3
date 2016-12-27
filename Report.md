# 魏凱亞 、劉祐欣 、柯子逸 、林怡均 <span style="color:red">(105062504、105062536、105065514、105062518)</span>

#Homework 3 / Generative Models

## Overview
The project is related to 
> Generative models、Deep Convolutional Generative Adversarial Networks

The code of this homework is from: https://github.com/carpedm20/DCGAN-tensorflow

In this project, our goal is to train a generative model that can generate real images that would looked just the same as natural images. Each team would submit 500 generated images, then all the images are combined with some real images which become the validation data for all the teams.



## Implementation
### Ver.1 Change the output size to 256*256 directly
 
In version 1, we only change the output size to 256. This means the architecture of the model and amount of parameters are the same as the original model.
We simply modify the default value in "main.py".

```
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
```
The outcome shows a not so bad result. The images all looks different to each other. This means that the model parameters did learn to generate multiple real images.


### Ver.2 Add another 2 convolutional layers(and 2 deconv. layers)

In version 2, We want to analyze the results if the additional layers added to the model would be a good choice. Because of that we want to generate a larger images of size 256 rather then 64. We thought that adding more layers might improve the visualized image results.
There are 3 parts to be modified: Generator、Discriminator、Sampler.



The Generator defines the deconvolution network that would take a noise input Z, and output a images of size 256*256.

Generator
```
 def generator(self, z, y=None):
            s = self.output_size
            s2, s4, s8, s16 , s32 , s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32) , int(s/64)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*32*s64*s64, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s64, s64, self.gf_dim * 32])
            h0 = tf.nn.relu(self.g_bn0(self.h0))


            self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                [self.batch_size, s32, s32, self.gf_dim*16], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_h4', with_w=True)
            h4 = tf.nn.relu(self.g_bn4(h4))

            h5, self.h5_w, self.h5_b = deconv2d(h4,
                [self.batch_size, s2, s2, self.gf_dim*1], name='g_h5', with_w=True)
            h5 = tf.nn.relu(self.g_bn5(h5))

            h6, self.h6_w, self.h6_b = deconv2d(h5,
                [self.batch_size, s, s, self.c_dim], name='g_h6', with_w=True)

            return tf.nn.tanh(h6)
```


The Discriminator defines the convolutional network which take a image as input(in training, the input would be the output from generator), and then output the score whether the image is real or fake image.


Discriminator
```
 def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*16, name='d_h4_conv')))
        h5 = lrelu(self.d_bn5(conv2d(h4, self.df_dim*32, name='d_h5_conv')))
        h6 = linear(tf.reshape(h5, [self.batch_size, -1]), 1, 'd_h5_lin')

        return tf.nn.sigmoid(h4), h4
```


The Sampler is used when we want to viusualize the output result.

Sampler
```
def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()
        s = self.output_size
        s2, s4, s8, s16 ,s32,s64= int(s/2), int(s/4), int(s/8), int(s/16), int(s/32) , int(s/64)

        # project `z` and reshape
        h0 = tf.reshape(linear(z, self.gf_dim*32*s64*s64, 'g_h0_lin'),
                        [-1, s64, s64, self.gf_dim * 32])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s32, s32, self.gf_dim*16], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s16, s16, self.gf_dim*8], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h4')
        h4 = tf.nn.relu(self.g_bn4(h4, train=False))

        h5 = deconv2d(h4, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h5')
        h5 = tf.nn.relu(self.g_bn5(h5, train=False))

        h6 = deconv2d(h5, [self.batch_size, s, s, self.c_dim], name='g_h6')

        return tf.nn.tanh(h6)

```

### Test images using Discriminator

If one want to use the Discriminator to distinguish whether the given image is a real or fake one. The implemented function "predict_result" can be used. The function take a input consist of 8 images each row and column. For each input, it would output 2 64*1 score map which is the score of positive and negative of each original image.

```
def predict_result(sess, dcgan, config):
  batch_images = np.zeros([64,256,256,3])
  count=1
  while count<2337:
    for i in range(count,count+63,1):
      if i < 9:
        filename = ('/data/mplab105/homework3//DCGAN-tensorflow/data/eval_data/000%d.jpg' % (i+1))
      elif i < 99:
        filename = ('/data/mplab105/homework3//DCGAN-tensorflow/data/eval_data/00%d.jpg' % (i+1))
      elif i < 999:
        filename = ('/data/mplab105/homework3//DCGAN-tensorflow/data/eval_data/0%d.jpg' % (i+1))
      else:
        filename = ('/data/mplab105/homework3//DCGAN-tensorflow/data/eval_data/%d.jpg' % (i+1))
      batch = imread(filename)
      batch = np.array(batch).astype(np.float32)
      batch_images[i%64,...] = batch
    x_placeholder = tf.placeholder(tf.float32,shape = batch_images.shape,name = 'input')
    logit = sess.run(dcgan.discriminator(x_placeholder, reuse=True), feed_dict={x_placeholder:batch_images})
    np.savetxt('predict%d.txt' % (count/64) ,logit,fmt = '%f\n')
    count+=64
```


## Discussion
Although we thought that adding 2 layers would improve performance, The result are pretty bad. All the images tend to look similar to each other. There might be only 4 different scenes out of 64 images. Most of the images looks the same except for some minor color difference.
The submitted images and the discrimia=nator we used are from version 1. We thought that there are too many parameters from version 2 that the generator take all the input to the best, one and only one vector space.  

##Visualized Results
<table border=1>
<tr>
<td>
<img src="images/o1.png" width="24%"/>
<img src="images/o2.png" width="24%"/>
<img src="images/o3.png" width="24%"/>
<img src="images/o4.png" width="24%"/>
</td>
</tr>

<tr>
<td>
<img src="images/a1.png" width="24%"/>
<img src="images/a1.png" width="24%"/>
<img src="images/a1.png" width="24%"/>
<img src="images/a1.png" width="24%"/>
</td>
</tr>

</table>

## Contribution
魏凱亞:discussion、code editing、submit result、report writing
劉祐欣:discussion、report writing
柯子逸:discussion、report writing
林怡均:discussion、report writing
