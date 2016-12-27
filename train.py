import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
from Utils import image_processing
import scipy.misc
import random
import json
import os
import shutil
from scipy.misc import imsave
import subprocess

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', type=str, default="train",
                        help='train, test')

    parser.add_argument('--z_dim', type=int, default=100,
                        help='Noise dimension')

    parser.add_argument('--s_dim', type=int, default=256,
                        help='Sound feature dimension')

    parser.add_argument('--batch_size', type=int, default=80,
                        help='Batch Size')

    parser.add_argument('--image_size', type=int, default=64,
                        help='Image Size a, a x a')

    parser.add_argument('--image_c_dim', type=int, default=3,
                        help='Image Channel Dimension')

    parser.add_argument('--gf_dim', type=int, default=64,
                        help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64,
                        help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024,
                        help='Dimension of gen untis for for fully connected layer 1024')

#    parser.add_argument('--sound_c_dim', type=int, default=1024,
#                        help='Sound channel dimension')

#    parser.add_argument('--sound_length', type=int, default=8,
#                        help='Sound length')

    parser.add_argument('--data_dir', type=str, default="Data",
                        help='Data Directory')

    parser.add_argument('--save_dir', type=str, default="Models_stack256_80",
                        help='Models Directory')

    parser.add_argument('--sample_dir', type=str, default="Samples_inference_80",
                        help='Samples Directory')

    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Momentum for Adam Update')

    parser.add_argument('--epochs', type=int, default=600,
                        help='Max number of epochs')

    parser.add_argument('--save_every', type=int, default=200,
                        help='Save Model/Samples every x iterations over batches')

    parser.add_argument('--model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--dataset', type=str, default="CIFAR10",
                        help='Data set: ESC10')

    args = parser.parse_args()

    model_options = {
        'z_dim' : args.z_dim,
        's_dim' : args.s_dim,
        'batch_size' : args.batch_size,
        'image_size' : args.image_size,
        'image_c_dim' : args.image_c_dim,
        'gf_dim' : args.gf_dim,
        'df_dim' : args.df_dim,
        'gfc_dim' : args.gfc_dim,
#        'sound_length' : args.sound_length,
#        'sound_c_dim' : args.sound_c_dim
    }

    if not os.path.exists(args.save_dir):
        subprocess.call('mkdir %s'%(args.save_dir),shell=True)
    if not os.path.exists(args.sample_dir):
        subprocess.call('mkdir %s'%(args.sample_dir),shell=True)

    log_dir = './logs_256_80'
    if not os.path.exists(log_dir):
        subprocess.call('mkdir %s'%(log_dir),shell=True)

    gan = model.GAN(model_options)
    input_tensors, input_tensors_z, variables, loss, outputs, checks = gan.build_model()
    
    global_step = tf.Variable(0, trainable = False)
    starter_learning_rate = args.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 600*50000*3, 0.5, staircase = True)
    #Try to merge summaries
    merged = tf.merge_all_summaries()

    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])

    d2_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1).minimize(loss['d2_loss'], var_list=variables['d2_vars'])
    g2_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1).minimize(loss['g2_loss'], var_list=variables['g2_vars'])

    room_data = np.load('indoor_data_64.npy')
#    room_data = (room_data/(255.0/2.0))-1.0
   
    room_data_2 = np.load('indoor_data_256.npy') 
#    room_data_2 = (room_data_2/(255.0/2.0))-1.0

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)
        tf.initialize_all_variables().run()
       # var_list_1 = [v for v in variables['g_vars']]
       # var_list_2 = [v for v in variables['d_vars']]
       # var_list_restore = sum([var_list_1, var_list_2], [])
       # saver = tf.train.Saver(var_list_restore)
        saver = tf.train.Saver(max_to_keep=100)
        if args.model:
            saver.restore(sess, args.model)
            
        saver = tf.train.Saver(max_to_keep=None)
        
        batch_image = np.zeros((model_options['batch_size'], 64, 64, 3))
        batch_image_2 = np.zeros((model_options['batch_size'], 256, 256, 3))
        batch_label = np.zeros((model_options['batch_size'], 10))
        batch_label_index = np.zeros(model_options['batch_size'])
        batch_z = np.zeros((model_options['batch_size'], 100))
        
        for epoch in range(args.epochs):
            # shuffle batch indexes in next epoch

            batch_idx = 0
            while batch_idx < int(50000/args.batch_size):
                batch_z = np.random.uniform(-1, 1, (model_options['batch_size'], 100))
                for class_idx in range(10):
                    image_per_class = model_options['batch_size']/10
                   # print image_per_class
                    class_list = np.random.randint(5000, size = image_per_class) 
                    batch_image[(class_idx)*image_per_class:(class_idx+1)*image_per_class, :, :, :] = ((room_data[class_idx, class_list, :, :, :])/(255.0/2.0)) - 1.0
                    batch_image_2[(class_idx)*image_per_class:(class_idx+1)*image_per_class, :, :, :] = ((room_data_2[class_idx, class_list, :, :, :])/(255.0/2.0)) - 1.0
                for _ in range(model_options['batch_size']):
                   # print _
                   # print batch_image[_,:,:,:]
                    assert np.array_equal(batch_image[_,:,:,:],np.zeros((64,64,3)))==False
                    assert np.array_equal(batch_image_2[_,:,:,:],np.zeros((256,256,3)))==False

                real_images = batch_image
                real_images_2 = batch_image_2
		# Update D
                """
#                d_losses = [ checks['d_loss1'] , checks['d_loss2']]
                _, d_loss, gen = sess.run([d_optim, loss['d_loss'], outputs['generator']], feed_dict = {input_tensors_z['t_z'] : batch_z, input_tensors['t_real_image'] : real_images})
                # Update G
                _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']], feed_dict = {input_tensors_z['t_z'] : batch_z, input_tensors['t_real_image'] : real_images})
                # Update G twice, to make sure d_loss does not go to 0
                _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']], feed_dict = {input_tensors_z['t_z'] : batch_z, input_tensors['t_real_image'] : real_images})
                
                print "Epoch: [{}] Batch: {} D_loss:{:5f} G_loss: {:5f}".format( epoch, batch_idx, d_loss, g_loss)
                """

                # Update D2
                _, d2_loss, gen_2 = sess.run([d2_optim, loss['d2_loss'], outputs['generator_2']], feed_dict = { input_tensors_z['t_z'] : batch_z, input_tensors['t_real_image_2'] : real_images_2})
                # Update G2
                _, g2_loss, gen_2 = sess.run([g2_optim, loss['g2_loss'], outputs['generator_2']], feed_dict = {input_tensors_z['t_z'] : batch_z, input_tensors['t_real_image_2'] : real_images_2})
#                # Update G2 twice, to make sure d_loss does not go to 0
#                _, g2_loss, gen_2 = sess.run([g2_optim, loss['g2_loss'], outputs['generator_2']], feed_dict = { input_tensors_z['t_z'] : batch_z, input_tensors['t_real_image_2'] : real_images_2})

                print "Epoch: [{}] Batch: {} D_loss:{:5f} G_loss: {:5f}".format( epoch, batch_idx, d2_loss, g2_loss)
                
                batch_idx += 1
                if (batch_idx % 40) == 0:
                    print "Saving Images"
#                    save_samples(args.sample_dir, gen, epoch, batch_idx, stage = "I")#, true_digits)
                    save_samples(args.sample_dir, gen_2, epoch, batch_idx, stage = "II")
            	if (batch_idx %args.save_every) == 0:
                	save_path = saver.save(sess, args.save_dir + "/model_epoch{}_batch_{}.ckpt".format(epoch, batch_idx))
                if batch_idx %20 ==0:
                    result=sess.run(merged,feed_dict = { input_tensors_z['t_z'] : batch_z, input_tensors['t_real_image_2'] : real_images_2,input_tensors['t_real_image'] : real_images})
                    summary_writer.add_summary(result,batch_idx+int(50000/args.batch_size)*epoch)

def save_samples(sample_dir, generated_images, epoch, batch_idx, stage = None): #, true_digits):
    for _ in range(len(generated_images)):
        image = generated_images[_]
#       true_digit = true_digits[0]
        if stage == "I":
            image = np.reshape(image, [64, 64, 3])
        elif stage == "II":
            image = np.reshape(image, [256, 256, 3])

        image = (image+1.0)*(255.0/2.0)
        image = np.array(image, dtype='uint8')
        imsave(sample_dir + '/epoch_{}_batch{}_{}'.format(epoch, batch_idx,_) + '.png', image)


if __name__ == '__main__':
	main()
