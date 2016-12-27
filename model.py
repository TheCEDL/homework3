import tensorflow as tf
from Utils import ops
import numpy as np
class GAN:
    '''
    OPTIONS
    z_dim : Noise dimension 100
    s_dim : Sound feature dimension 256
    image_size : Image Dimension 64
    image_c_dim: Image Channel Dimension 1
    gf_dim : Number of conv in the first layer generator 64
    df_dim : Number of conv in the first layer discriminator 64
    gfc_dim : Dimension of gen untis for for fully connected layer 1024
    sound__length : Sound length 8
    sound__c_dim : Sound channel dim 1024
    batch_size : Batch Size 64
    '''
    def __init__(self, options):
        self.options = options

        # Used for sound embedding : No need Now!!
        self.g_s_bn0 = ops.batch_norm(name='g_s_bn0')
        self.g_s_bn1 = ops.batch_norm(name='g_s_bn1')
        self.g_s_bn2 = ops.batch_norm(name='g_s_bn2')
        self.d_s_bn0 = ops.batch_norm(name='d_s_bn0')
        self.d_s_bn1 = ops.batch_norm(name='d_s_bn1')
        self.d_s_bn2 = ops.batch_norm(name='d_s_bn2')

        # Used for noise vector
        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')

        self.d_bn1 = ops.batch_norm(name='d_bn1')
        self.d_bn2 = ops.batch_norm(name='d_bn2')
        self.d_bn3 = ops.batch_norm(name='d_bn3')
        self.d_bn4 = ops.batch_norm(name='d_bn4')
        self.d_bn5 = ops.batch_norm(name='d_bn5')

        self.down_bn1 = ops.batch_norm(name='down_bn1')
        
        #The batch normalization layers of the residual blocks
        self.r_bn0  = ops.batch_norm(name='res_bn0')
        self.r_bn1  = ops.batch_norm(name='res_bn1')
        self.r_bn3  = ops.batch_norm(name='res_bn3')
        self.r_bn4  = ops.batch_norm(name='res_bn4')
        self.r_bn6  = ops.batch_norm(name='res_bn6')
        self.r_bn7  = ops.batch_norm(name='res_bn7')
        self.r_bn9  = ops.batch_norm(name='res_bn9')
        self.r_bn10 = ops.batch_norm(name='res_bn10')

        self.g2_bn0 = ops.batch_norm(name='g2_bn0')
        self.g2_bn1 = ops.batch_norm(name='g2_bn1')
        self.g2_bn2 = ops.batch_norm(name='g2_bn2')

        self.d2_bn1 = ops.batch_norm(name='d2_bn1')
        self.d2_bn2 = ops.batch_norm(name='d2_bn2')
        self.d2_bn3 = ops.batch_norm(name='d2_bn3')
        self.d2_bn4 = ops.batch_norm(name='d2_bn4')

        #Some summary histogram of the training


    def build_model(self):
        img_size = self.options['image_size']
        img_c_dim = self.options['image_c_dim']
        t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, img_c_dim], name = 'real_image')
        t_real_image_2 = tf.placeholder('float32', [self.options['batch_size'],img_size*4, img_size*4, img_c_dim], name = 'real_image_2')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
#        batch_labels_idx = tf.placeholder('int32', [self.options['batch_size']])		
            
        # Stage 1 Generator ######################
        fake_image = self.generator(t_z)
        
        # Stage 2 Generator ######################
        down_sample = self.downsampling(fake_image)
        residue = self.residuel_block(down_sample)
        fake_image_2 = self.generator_2(residue)
        self.generated_image = tf.image_summary("image256",fake_image_2,max_images=10)
	
        # Stage 1&2 Discriminator real image##################
        disc_real_image, disc_real_image_logits = self.discriminator(t_real_image)
        disc_real_image_2, disc_real_image_logits_2 = self.discriminator_2(t_real_image_2)
        
        # Stage 1&2 Discriminator fake image#################
        disc_fake_image, disc_fake_image_logits = self.discriminator(fake_image, reuse = True)
        disc_fake_image_2, disc_fake_image_logits_2 = self.discriminator_2(fake_image_2)

        # Stage1 generator loss##################
        g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( disc_fake_image_logits, tf.ones_like(disc_fake_image)))
        self.g_loss_1 = tf.scalar_summary("generator_loss_1",g_loss1)

        # Stage1 discriminator loss##############
        d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( disc_real_image_logits, tf.ones_like(disc_real_image)))
        self.d_loss_real = tf.scalar_summary("discriminator_loss_real_1",d_loss1)

        d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( disc_fake_image_logits, tf.zeros_like(disc_fake_image)))
        self.d_loss_fake = tf.scalar_summary("discriminator_loss_fake_1",d_loss2)

        # Stage1 loss ###########################
        d_loss = d_loss1 + d_loss2 #+ rec_loss1 + rec_loss2
        self.d_loss_total = tf.scalar_summary("discriminator_loss_total_1",d_loss)

        g_loss = g_loss1 #+ rec_loss1 + rec_loss2
        
        # Stage2 generator loss##################
        g2_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( disc_fake_image_logits_2, tf.ones_like(disc_fake_image_2)))

        # Stage2 discriminator loss##############
        d2_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( disc_real_image_logits_2, tf.ones_like(disc_real_image_2)))
        self.d_loss_real_2 = tf.scalar_summary("discriminator_loss_real_2",d2_loss1)

        d2_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( disc_fake_image_logits_2, tf.zeros_like(disc_fake_image_2)))
        self.d_loss_fake_2 = tf.scalar_summary("discriminator_loss_fake_2",d2_loss2)
        # Stage2 loss############################
        d2_loss = d2_loss1 + d2_loss2
        self.d_loss_total_2 = tf.scalar_summary("discriminator_loss_total_2",d2_loss)

        g2_loss = g2_loss1 
        self.g_loss_2 = tf.scalar_summary("generator_loss_2",g2_loss)

        # Collect variable ############
        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

#        down_vars = [var for var in t_vars if 'down_' in var.name]
#        res_vars = [var for var in t_vars if 'res' in var.name]
        d2_vars = [var for var in t_vars if 'd2_' in var.name]

        g2_vars_ = [var for var in t_vars if 'g2_' in var.name]
        down_vars_ = [var for var in t_vars if 'down_' in var.name]
        res_vars_ = [var for var in t_vars if 'res_' in var.name]

        g2_vars = sum([g2_vars_, down_vars_, res_vars_], []) 

        # Model IO port ###############
        input_tensors = {
            't_real_image' : t_real_image,
            't_real_image_2' : t_real_image_2
        }
        input_tensors_z = {
            't_z' : t_z
        }
        variables = {
            'd_vars' : d_vars,
            'g_vars' : g_vars,
#            'down_vars' : down_vars,
#            'res_vars' : res_vars,
            'd2_vars' : d2_vars,
            'g2_vars' : g2_vars,
        }

        loss = {
            'g_loss' : g_loss,
            'd_loss' : d_loss,
            'g2_loss': g2_loss,
            'd2_loss': d2_loss,
        }

        outputs = {
            'generator' : fake_image,
            'generator_2' : fake_image_2
        }

        checks = {
            'disc_real_image_real_sound_logits' : disc_real_image_2,
            'disc_fake_image_real_sound_logits' : disc_real_image_logits_2
        }

        return input_tensors, input_tensors_z, variables, loss, outputs, checks


   # GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def generator(self, t_z): #, t_sound_embedding):
        s = self.options['image_size']
        
        # Fully connected layer with feature map 4*4*512
        z_ = ops.linear(t_z, 64*8*4*4, 'g_h0_lin')
        h0 = tf.reshape(z_, [-1, 4, 4, 64*8])
        h0 = tf.nn.relu(self.g_bn0(h0))
		
        # deconvolution with feature map 8*8*256
        h1 = ops.deconv2d(h0, [self.options['batch_size'], 8, 8, 64*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1))
        
        # deconvolution with feature map 16*16*128
        h2 = ops.deconv2d(h1, [self.options['batch_size'], 16, 16, 64*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2))

        # deconvolution with feature map 32*32*64
        h3 = ops.deconv2d(h2, [self.options['batch_size'], 32, 32, 64*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3))

        # deconvolution with feature map 64*64*3
        h4 = ops.deconv2d(h3, [self.options['batch_size'], 64, 64, self.options['image_c_dim']], name='g_h4')
        
        # tanh is the same setting from DCGAN
        return (tf.tanh(h4))


   # DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = ops.lrelu( ops.conv2d(image, 64, 
                        name = 'd_h0_conv')) #32
        h1 = ops.lrelu( self.d_bn1(ops.conv2d(h0, 128, 
                        name = 'd_h1_conv'))) #32
        h2 = ops.lrelu( self.d_bn2(ops.conv2d(h1, 256, 
                        name = 'd_h2_conv'))) #32
        h3 = ops.lrelu( self.d_bn3(ops.conv2d(h2, 512, 
                        name = 'd_h3_conv'))) #32
	
        h4 = ops.linear(tf.reshape(h3, [self.options['batch_size'], -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4 

    def downsampling(self, image, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        # Downsample to 32*32*256
        h0 = ops.lrelu( ops.conv2d(image, 256, 
                        name = 'down_h0_conv')) #32

        # Downsample to 16*16*512
        h1 = ops.lrelu( self.down_bn1(ops.conv2d(h0, 512, 
                        name = 'down_h1_conv'))) #32
	return h1

    def residuel_block(self, input_tensor, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        #[To do] it seemed that after identity mapping, there should be a relu
        h0 = tf.nn.relu(self.r_bn0(ops.conv2d(input_tensor, 512, 
                                   k_h=3, k_w=3, d_h=1, d_w=1,
                                   name = 'res_h0')))
        h1 = self.r_bn1(ops.conv2d(h0, 512, 
                                   k_h=3, k_w=3, d_h=1, d_w=1, 
                                   name = 'res_h1'))
        h2 = input_tensor + h1

        h3 = tf.nn.relu(self.r_bn3(ops.conv2d(h2, 512, 
                                   k_h=3, k_w=3, d_h=1, d_w=1,
                                   name = 'res_h3')))
        h4 = self.r_bn4(ops.conv2d(h3, 512, 
                                   k_h=3, k_w=3, d_h=1, d_w=1, 
                                   name = 'res_h4'))
        h5 = h4 + h2
        
        h6 = tf.nn.relu(self.r_bn6(ops.conv2d(h5, 512, 
                                   k_h=3, k_w=3, d_h=1, d_w=1,
                                   name = 'res_h6')))
        h7 = self.r_bn7(ops.conv2d(h6, 512, 
                                   k_h=3, k_w=3, d_h=1, d_w=1, 
                                   name = 'res_h7'))
        h8 = h7 + h5
 
        h9 = tf.nn.relu(self.r_bn9(ops.conv2d(h8, 512, 
                                   k_h=3, k_w=3, d_h=1, d_w=1,
                                   name = 'res_h9')))
        h10 = self.r_bn10(ops.conv2d(h9, 512, 
                                   k_h=3, k_w=3, d_h=1, d_w=1, 
                                   name = 'res_h10'))
        h8 = h10 + h8

        return h8

    def generator_2(self, input_tensor, reuse = False): #, t_sound_embedding):
        if reuse:
            tf.get_variable_scope().reuse_variables()
		
        h0 = ops.deconv2d(input_tensor, [self.options['batch_size'], 32, 32, 256], name='g2_h0')
        h0 = tf.nn.relu(self.g2_bn0(h0))

        h1 = ops.deconv2d(h0, [self.options['batch_size'], 64, 64, 128], name='g2_h1')
        h1 = tf.nn.relu(self.g2_bn1(h1))

        h2 = ops.deconv2d(h1, [self.options['batch_size'], 128, 128, 64], name='g2_h2')
        h2 = tf.nn.relu(self.g2_bn2(h2))

        h3 = ops.deconv2d(h2, [self.options['batch_size'], 256, 256, 3], name='g2_h3')

        return (tf.tanh(h3))

   
    def discriminator_2(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = ops.lrelu( ops.conv2d(image, 64, 
                        name = 'd2_h0_conv')) #32
        h1 = ops.lrelu( self.d2_bn1(ops.conv2d(h0, 128, 
                        name = 'd2_h1_conv'))) #32
        h2 = ops.lrelu( self.d2_bn2(ops.conv2d(h1, 256, 
                        name = 'd2_h2_conv'))) #32
        h3 = ops.lrelu( self.d2_bn3(ops.conv2d(h2, 512, 
                        name = 'd2_h3_conv'))) #32
        h4 = ops.lrelu( self.d2_bn4(ops.conv2d(h3, 1024, 
                        name = 'd2_h4_conv'))) #32
	
        h5 = ops.linear(tf.reshape(h4, [self.options['batch_size'], -1]), 1, 'd2_h5_lin')

        return tf.nn.sigmoid(h5), h5
