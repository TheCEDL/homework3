from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from scipy.misc import imsave
from ops import *
from utils import *

import pdb

class DCGAN(object):
    def __init__(self, sess, phase='2', image_size=108, is_crop=True,
                 batch_size=64, sample_size = 64, output_size=64,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
	self.dataset_name = 'place'
	self.phase  = phase
        self.is_crop = is_crop
        self.is_grayscale = False
        self.batch_size = batch_size
        self.image_size = 256
        self.sample_size = sample_size
        self.output_size_1 = 64
	self.output_size_2 = 256

        self.z_dim = z_dim

        self.gf_dim = gf_dim
	self.gf_dim_2 = 64
        self.df_dim = df_dim
	self.df_dim_2 = 256 

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d1_bn1 = batch_norm(name='d1_bn1')
        self.d1_bn2 = batch_norm(name='d1_bn2')
        self.d1_bn3 = batch_norm(name='d1_bn3')

        self.g1_bn0 = batch_norm(name='g1_bn0')
        self.g1_bn1 = batch_norm(name='g1_bn1')
        self.g1_bn2 = batch_norm(name='g1_bn2')
        self.g1_bn3 = batch_norm(name='g1_bn3')
	
	self.d2_bn1 = batch_norm(name='d2_bn1')
	self.d2_bn2 = batch_norm(name='d2_bn2')
	self.d2_bn3 = batch_norm(name='d2_bn3')
	self.d2_bn4 = batch_norm(name='d2_bn4')
	self.d2_bn5 = batch_norm(name='d2_bn5')

	self.g2_bn1 = batch_norm(name='g2_bn1')	
	self.g2_bn2 = batch_norm(name='g2_bn2')
	self.g2_bn3 = batch_norm(name='g2_bn3')
	self.g2_bn4 = batch_norm(name='g2_bn4')

	self.r_bn0 = batch_norm(name='g2_r_bn0')
	self.r_bn1 = batch_norm(name='g2_r_bn1')
	self.r_bn3 = batch_norm(name='g2_r_bn3')
	self.r_bn4 = batch_norm(name='g2_r_bn4')
	self.r_bn6 = batch_norm(name='g2_r_bn6')
	self.r_bn7 = batch_norm(name='g2_r_bn7')
	self.r_bn9 = batch_norm(name='g2_r_bn9')
	self.r_bn10 = batch_norm(name='g2_r_bn10')	

        self.checkpoint_dir = checkpoint_dir
	self.checkpoint_path = './checkpoint/place_64_1/DCGAN.model-50'
        self.build_model()

    def build_model(self):
        self.images1 = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size_1, self.output_size_1, self.c_dim], name='real_images1')
	self.images2 = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size_2, self.output_size_2, self.c_dim], name='real_images2')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
	self.z_sum = tf.histogram_summary("z", self.z)


        self.G1 = self.generator1(self.z)
        self.D1, self.D1_logits  = self.discriminator1(self.images1, reuse=False)
        self.D1_, self.D1_logits_ = self.discriminator1(self.G1, reuse=True)

        self.G2 = self.generator2(self.G1)
        self.D2, self.D2_logits = self.discriminator2(self.images2, reuse=False)
        self.D2_, self.D2_logits_ = self.discriminator2(self.G2, reuse=True)

	self.sampler1 = self.sampler1(self.z)        
	self.sampler2 = self.sampler2(self.sampler1)

        self.d1_sum = tf.histogram_summary("d1", self.D1)
        self.d1__sum = tf.histogram_summary("d1_", self.D1_)
        self.G1_sum = tf.image_summary("G1", self.G1)

        self.d2_sum = tf.histogram_summary("d2", self.D2)
        self.d2__sum = tf.histogram_summary("d2_", self.D2_)
        self.G2_sum = tf.image_summary("G2", self.G2)

        self.d1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits, tf.ones_like(self.D1)))
        self.d1_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits_, tf.zeros_like(self.D1_)))
        self.g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D1_logits_, tf.ones_like(self.D1_)))

        self.d2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits, tf.ones_like(self.D2)))
        self.d2_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.zeros_like(self.D2_)))
        self.g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.ones_like(self.D2_)))

        self.d1_loss_real_sum = tf.scalar_summary("d1_loss_real", self.d1_loss_real)
        self.d1_loss_fake_sum = tf.scalar_summary("d1_loss_fake", self.d1_loss_fake)

        self.d2_loss_real_sum = tf.scalar_summary("d2_loss_real", self.d2_loss_real)
        self.d2_loss_fake_sum = tf.scalar_summary("d2_loss_fake", self.d2_loss_fake)
                                                    
        self.d1_loss = self.d1_loss_real + self.d1_loss_fake
	self.d2_loss = self.d2_loss_real + self.d2_loss_fake

        self.g1_loss_sum = tf.scalar_summary("g1_loss", self.g1_loss)
        self.d1_loss_sum = tf.scalar_summary("d1_loss", self.d1_loss)
        self.g2_loss_sum = tf.scalar_summary("g2_loss", self.g2_loss)
        self.d2_loss_sum = tf.scalar_summary("d2_loss", self.d2_loss)

        t_vars = tf.trainable_variables()

        self.d1_vars = [var for var in t_vars if 'd1_' in var.name]
	self.g1_vars = [var for var in t_vars if 'g1_' in var.name]
	self.d2_vars = [var for var in t_vars if 'd2_' in var.name]
	self.g2_vars = [var for var in t_vars if 'g2_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
	if self.phase == '1':
	    data = glob(os.path.join("/home/huchanwei123/StackGAN/Data/indoor_64", "*", "*.jpg"))
	else:
	    data = glob(os.path.join("/home/huchanwei123/StackGAN/Data/indoor", "*", "*.jpg"))

        d1_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d1_loss, var_list=self.d1_vars)
        g1_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g1_loss, var_list=self.g1_vars)

        d2_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d2_loss, var_list=self.d2_vars)
        g2_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g2_loss, var_list=self.g2_vars)

	tf.initialize_all_variables().run()

        self.g1_sum = tf.merge_summary([self.z_sum, self.d1__sum,
            self.G1_sum, self.d1_loss_fake_sum, self.g1_loss_sum])
        self.g2_sum = tf.merge_summary([self.z_sum, self.d2__sum,
            self.G2_sum, self.d2_loss_fake_sum, self.g2_loss_sum])

        self.d1_sum = tf.merge_summary([self.z_sum, self.d1_sum, self.d1_loss_real_sum, self.d1_loss_sum])
	self.d2_sum = tf.merge_summary([self.z_sum, self.d2_sum, self.d2_loss_real_sum, self.d2_loss_sum])

        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=False, resize_w=self.output_size_1, is_grayscale = self.is_grayscale) for sample_file in sample_files]
        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
            
        counter = 1
        start_time = time.time()

	if self.phase == '2':
            self.load(self.checkpoint_path)


        for epoch in xrange(config.epoch):
	    if self.phase == '1':
		data = glob(os.path.join("/home/huchanwei123/StackGAN/Data/indoor_64", "*", "*.jpg"))
	    else:
		data = glob(os.path.join("/home/huchanwei123/StackGAN/Data/indoor", "*", "*.jpg"))

            batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=False, resize_w=self.output_size_1, is_grayscale = self.is_grayscale) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

		if self.phase == '1':
                    # Update D network
                    _, summary_str = self.sess.run([d1_optim, self.d1_sum],
                        feed_dict={ self.images1: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g1_optim, self.g1_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g1_optim, self.g1_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)
                    
                    errD_fake = self.d1_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d1_loss_real.eval({self.images1: batch_images})
                    errG = self.g1_loss.eval({self.z: batch_z})
		else:
                    # Update D network
                    _, summary_str = self.sess.run([d2_optim, self.d2_sum],
                        feed_dict={ self.images2: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g2_optim, self.g2_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d2_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d2_loss_real.eval({self.images2: batch_images})
                    errG = self.g2_loss.eval({self.z: batch_z})


                counter += 1
		if idx % 200 == 2:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake+errD_real, errG))

	    if self.phase == '1':
	        if np.mod(epoch, 10) == 0:
		    self.save(config.checkpoint_dir, epoch)
	    if self.phase == '2':
		if np.mod(epoch, 1) == 0:
		    self.save(config.checkpoint_dir, epoch)


            if np.mod(epoch, 1) == 0:
		if self.phase == '1':
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler1, self.d1_loss, self.g1_loss],
                        feed_dict={self.z: sample_z, self.images1: sample_images}
                    )
		else:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler2, self.d2_loss, self.g2_loss],
                        feed_dict={self.z: sample_z, self.images2: sample_images}
                    )

                save_images(samples, [8, 8],
                            './{}/train_{:02d}.png'.format(config.sample_dir, epoch))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

    def sample(self, number):
	self.load(self.checkpoint_path)

	sample_z = np.random.uniform(-1, 1, size=(number, self.z_dim))
	samples = self.sess.run(
            self.G2,
            feed_dict={self.z: sample_z}
        )
	samples = np.reshape(samples, [number, 256, 256, 3])
	for i in range(number):
	    image = samples[i]
	    image = (image+1.0)*(255.0/2.0)
	    image = np.array(image, dtype='uint8')
	    imsave('final_samples/sample_{}.png'.format(i+1), image)	    

    def test(self, config):
        if self.phase == '1':
            data = glob(os.path.join("/data2/Places2_subset/flatten64", "*.jpg"))
        else:
            data = glob(os.path.join("/data2/Places2_subset/flatten", "*.jpg"))

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=False, resize_w=self.output_size_1, is_grayscale = self.is_grayscale) for sample_file in sample_files]
        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        samples, d_loss, g_loss = self.sess.run(
            [self.sampler1, self.d1_loss, self.g1_loss],
            feed_dict={self.z: sample_z, self.images1: sample_images}
        )
        save_images(samples, [8, 8],
                    './{}/test.png'.format(config.sample_dir))
        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))


    def discriminator1(self, image, reuse=False):
	with tf.variable_scope("d1"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d1_h0_conv'))
            h1 = lrelu(self.d1_bn1(conv2d(h0, self.df_dim*2, name='d1_h1_conv')))
            h2 = lrelu(self.d1_bn2(conv2d(h1, self.df_dim*4, name='d1_h2_conv')))
            h3 = lrelu(self.d1_bn3(conv2d(h2, self.df_dim*8, name='d1_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd1_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def discriminator2(self, image, reuse=False):
	with tf.variable_scope("d2"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d2_h0_conv'))		# 128
            h1 = lrelu(self.d2_bn1(conv2d(h0, self.df_dim*2, name='d2_h1_conv')))	# 62
            h2 = lrelu(self.d2_bn2(conv2d(h1, self.df_dim*4, name='d2_h2_conv')))	# 32
            h3 = lrelu(self.d2_bn3(conv2d(h2, self.df_dim*8, name='d2_h3_conv')))	# 16
	    h4 = lrelu(self.d2_bn4(conv2d(h3, self.df_dim*16, name='d2_h4_conv')))	# 8
            h5 = linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd2_h5_lin')

            return tf.nn.sigmoid(h5), h5

    def generator1(self, z):
	with tf.variable_scope("g1"):
            s = self.output_size_1
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g1_h0_lin', with_w=True)

            self.h0_1 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim*8])
            h0_1 = tf.nn.relu(self.g1_bn0(self.h0_1))

            self.h1_1, self.h1_w_1, self.h1_b_1 = deconv2d(h0_1, [self.batch_size, s8, s8, self.gf_dim*4], name='g1_h1', with_w=True)
            h1_1 = tf.nn.relu(self.g1_bn1(self.h1_1))

            h2_1, self.h2_w_1, self.h2_b_1 = deconv2d(h1_1, [self.batch_size, s4, s4, self.gf_dim*2], name='g1_h2', with_w=True)
            h2_1 = tf.nn.relu(self.g1_bn2(h2_1))

            h3_1, self.h3_w_1, self.h3_b_1 = deconv2d(h2_1, [self.batch_size, s2, s2, self.gf_dim*1], name='g1_h3', with_w=True)
            h3_1 = tf.nn.relu(self.g1_bn3(h3_1))

            h4_1, self.h4_w_1, self.h4_b_1 = deconv2d(h3_1, [self.batch_size, s, s, self.c_dim], name='g1_h4', with_w=True)
            return tf.nn.tanh(h4_1)

    def generator2(self, image):
        with tf.variable_scope("g2"):
	    s = self.output_size_2
	    s2, s4, s8 = int(s/2), int(s/4), int(s/8)

            h0 = lrelu(conv2d(image, self.df_dim_2, name='g2_h0_conv'))		# [32, 32, 256]
            h1 = lrelu(self.g2_bn1(conv2d(h0, self.df_dim_2*2, name='g2_h1_conv')))	# [16, 16, 512]

	    h1_res = self.residual_block(h1)

            self.h2_2, self.h2_w_2, self.h2_b_2 = deconv2d(h1_res, [self.batch_size, s8, s8, self.gf_dim_2*4], name='g2_h2', with_w=True)
            h2_2 = tf.nn.relu(self.g2_bn2(self.h2_2))	# [32, 32, 256]

            h3_2, self.h3_w_2, self.h3_b_2 = deconv2d(h2_2, [self.batch_size, s4, s4, self.gf_dim_2*2], name='g2_h3', with_w=True)
            h3_2 = tf.nn.relu(self.g2_bn3(h3_2))	# [64, 64, 128]

            h4_2, self.h4_w_2, self.h4_b_2 = deconv2d(h3_2, [self.batch_size, s2, s2, self.gf_dim_2*1], name='g2_h4', with_w=True)
            h4_2 = tf.nn.relu(self.g2_bn4(h4_2))        # [128, 128, 64]

            h5_2, self.h5_w_2, self.h5_b_2 = deconv2d(h4_2, [self.batch_size, s, s, self.c_dim], name='g2_h5', with_w=True)
	    return tf.nn.tanh(h5_2)            	# [256, 256, 3]

    def residual_block(self, input_tensor, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = tf.nn.relu(self.r_bn0(conv2d(input_tensor, 512, k_h=3, k_w=3, d_h=1, d_w=1, name = 'g2_res_h0')))
        h1 = self.r_bn1(conv2d(h0, 512, k_h=3, k_w=3, d_h=1, d_w=1, name = 'g2_res_h1'))
        h2 = input_tensor + h1

        h3 = tf.nn.relu(self.r_bn3(conv2d(h2, 512, k_h=3, k_w=3, d_h=1, d_w=1, name = 'g2_res_h3')))
        h4 = self.r_bn4(conv2d(h3, 512, k_h=3, k_w=3, d_h=1, d_w=1, name = 'g2_res_h4'))
        h5 = h4 + h2

        h6 = tf.nn.relu(self.r_bn6(conv2d(h5, 512, k_h=3, k_w=3, d_h=1, d_w=1, name = 'g2_res_h6')))
        h7 = self.r_bn7(conv2d(h6, 512, k_h=3, k_w=3, d_h=1, d_w=1, name = 'g2_res_h7'))
        h8 = h7 + h5

        h9 = tf.nn.relu(self.r_bn9(conv2d(h8, 512, k_h=3, k_w=3, d_h=1, d_w=1, name = 'g2_res_h9')))
        h10 = self.r_bn10(conv2d(h9, 512, k_h=3, k_w=3, d_h=1, d_w=1, name = 'g2_res_h10'))
        h11 = h10 + h8

        return h11

    def sampler1(self, z):
        with tf.variable_scope("g1"):
	    tf.get_variable_scope().reuse_variables()
            s = self.output_size_1
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g1_h0_lin', with_w=True)

            self.h0_1 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim*8])
            h0_1 = tf.nn.relu(self.g1_bn0(self.h0_1))

            self.h1_1, self.h1_w_1, self.h1_b_1 = deconv2d(h0_1, [self.batch_size, s8, s8, self.gf_dim*4], name='g1_h1', with_w=True)
            h1_1 = tf.nn.relu(self.g1_bn1(self.h1_1))

            h2_1, self.h2_w_1, self.h2_b_1 = deconv2d(h1_1, [self.batch_size, s4, s4, self.gf_dim*2], name='g1_h2', with_w=True)
            h2_1 = tf.nn.relu(self.g1_bn2(h2_1))

            h3_1, self.h3_w_1, self.h3_b_1 = deconv2d(h2_1, [self.batch_size, s2, s2, self.gf_dim*1], name='g1_h3', with_w=True)
            h3_1 = tf.nn.relu(self.g1_bn3(h3_1))

            h4_1, self.h4_w_1, self.h4_b_1 = deconv2d(h3_1, [self.batch_size, s, s, self.c_dim], name='g1_h4', with_w=True)
            return tf.nn.tanh(h4_1)

    def sampler2(self, image):
        with tf.variable_scope("g2"):
	    tf.get_variable_scope().reuse_variables()
            s = self.output_size_2
            s2, s4, s8 = int(s/2), int(s/4), int(s/8)

            h0 = lrelu(conv2d(image, self.df_dim_2, name='g2_h0_conv'))         # [32, 32, 256]
            h1 = lrelu(self.g2_bn1(conv2d(h0, self.df_dim_2*2, name='g2_h1_conv')))     # [16, 16, 512]

            self.h2_2, self.h2_w_2, self.h2_b_2 = deconv2d(h1, [self.batch_size, s8, s8, self.gf_dim_2*4], name='g2_h2', with_w=True)
            h2_2 = tf.nn.relu(self.g2_bn2(self.h2_2))   # [32, 32, 256]

            h3_2, self.h3_w_2, self.h3_b_2 = deconv2d(h2_2, [self.batch_size, s4, s4, self.gf_dim_2*2], name='g2_h3', with_w=True)
            h3_2 = tf.nn.relu(self.g2_bn3(h3_2))        # [64, 64, 128]

            h4_2, self.h4_w_2, self.h4_b_2 = deconv2d(h3_2, [self.batch_size, s2, s2, self.gf_dim_2*1], name='g2_h4', with_w=True)
            h4_2 = tf.nn.relu(self.g2_bn4(h4_2))        # [128, 128, 64]

            h5_2, self.h5_w_2, self.h5_b_2 = deconv2d(h4_2, [self.batch_size, s, s, self.c_dim], name='g2_h5', with_w=True)
            return tf.nn.tanh(h5_2)             # [256, 256, 3]

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.phase)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            error_msg = "Can't find the checkpoint %s" % ckpt_path
            sys.exit(error_msg)
        else:
            try:
                self.saver.restore(self.sess, ckpt_path)
            except AttributeError:
                self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
                self.saver.restore(self.sess, ckpt_path)

    '''
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.phase)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    '''
