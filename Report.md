# Report

In this homework, I modified 3 files so that the model can generate images with 256 resolution.

## 1. main.py

flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [256]")

## 2. model.py

batch_size=64, sample_size = 64, output_size=256,
y_dim=None, z_dim=100, gf_dim=16, df_dim=16,

self.d_bn3 = batch_norm(name='d_bn3')
self.d_bn4 = batch_norm(name='d_bn4')

self.d_bn5 = batch_norm(name='d_bn5')

self.g_bn3 = batch_norm(name='g_bn3')
self.g_bn4 = batch_norm(name='g_bn4')

self.g_bn5 = batch_norm(name='g_bn5')

h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*16, name='d_h4_conv')))
h5 = lrelu(self.d_bn5(conv2d(h4, self.df_dim*32, name='d_h5_conv')))
h6 = linear(tf.reshape(h5, [self.batch_size, -1]), 1, 'd_h5_lin')

return tf.nn.sigmoid(h6), h6

s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64)

self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*32*s64*s64, 'g_h0_lin', with_w=True)

self.h0 = tf.reshape(self.z_, [-1, s64, s64, self.gf_dim * 32])

self.h1, self.h1_w, self.h1_b = deconv2d(h0,
[self.batch_size, s32, s32, self.gf_dim*16], name='g_h1', with_w=True)

h2, self.h2_w, self.h2_b = deconv2d(h1,
[self.batch_size, s16, s16, self.gf_dim*8], name='g_h2', with_w=True)

h3, self.h3_w, self.h3_b = deconv2d(h2,
[self.batch_size, s8, s8, self.gf_dim*4], name='g_h3', with_w=True)

h4, self.h4_w, self.h4_b = deconv2d(h3,
[self.batch_size, s4, s4, self.gf_dim*2], name='g_h4', with_w=True)
h4 = tf.nn.relu(self.g_bn4(h4))

h5, self.h5_w, self.h5_b = deconv2d(h4,
[self.batch_size, s2, s2, self.gf_dim*1], name='g_h5', with_w=True)
h5 = tf.nn.relu(self.g_bn5(h5))

h6, self.h6_w, self.h6_b = deconv2d(h5,
[self.batch_size, s, s, self.c_dim], name='g_h6', with_w=True)

return tf.nn.tanh(h6)

s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64)

h0 = tf.reshape(linear(z, self.gf_dim*32*s64*s64, 'g_h0_lin'),
[-1, s64, s64, self.gf_dim * 32])

h1 = deconv2d(h0, [self.batch_size, s32, s32, self.gf_dim*16], name='g_h1')

h2 = deconv2d(h1, [self.batch_size, s16, s16, self.gf_dim*8], name='g_h2')

h3 = deconv2d(h2, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h3')

h4 = deconv2d(h3, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h4')
h4 = tf.nn.relu(self.g_bn4(h4, train=False))

h5 = deconv2d(h4, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h5')
h5 = tf.nn.relu(self.g_bn5(h5, train=False))

h6 = deconv2d(h5, [self.batch_size, s, s, self.c_dim], name='g_h6')

return tf.nn.tanh(h6)

## 3. utils.py

def get_image(image_path, image_size, is_crop=True, resize_w=256, is_grayscale = False):

#for idx, image in enumerate(images):
#i = idx % size[1]
#j = idx // size[1]
#img[j*h:j*h+h, i*w:i*w+w, :] = image
img = images

def transform(image, npx=256, is_crop=True, resize_w=64):

for i in xrange(len(samples)):
save_images(samples[i], [1, 1], './samples/test_arange_%s_%s.png' % (idx, i))
