from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from ops import *
from utils import *
import math

class pix2pix(object):
    def __init__(self, sess, args):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.args = args
        self.is_grayscale = (self.args.input_nc == 1)
        self.batch_size = self.args.batch_size
        self.checkpoint_dir = self.args.checkpoint_dir
        self.model_name = 'dlmodel'
        self.output_size = self.args.fine_size
        self.data_dir = self.args.data_dir

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')
        self.g_bn_e9 = batch_norm(name='g_bn_e9')
        self.g_bn_e10 = batch_norm(name='g_bn_e10')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')
        self.g_bn_d8 = batch_norm(name='g_bn_d8')
        self.g_bn_d9 = batch_norm(name='g_bn_d9')

        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.args.batch_size, self.args.fine_size, self.args.fine_size,
                                         self.args.input_nc + self.args.output_nc],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.args.input_nc]
        self.real_B = self.real_data[:, :, :, self.args.input_nc:self.args.input_nc + self.args.output_nc]

        self.fake_B = self.generator(self.real_A)

        self.real_AB = tf.concat(3, [self.real_A, self.real_B])
        self.fake_AB = tf.concat(3, [self.real_A, self.fake_B])
        self.pos_logit, self.pos_vec = self.discriminator(self.real_AB, reuse=False)
        self.neg_logit, self.neg_vec = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)

        self.fake_B_sum = tf.image_summary("fake_B", self.fake_B)
        self.real_B_sum = tf.image_summary("real_B", self.real_B)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.pos_vec, tf.ones_like(self.pos_logit)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.neg_vec, tf.zeros_like(self.neg_logit)))

        self.g_loss = 0

        if self.args.use_gan:
            self.g_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.neg_vec, tf.ones_like(self.neg_logit)))
        if self.args.use_L1:
            self.g_loss += self.args.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def load_random_samples(self):
        data = np.random.choice(glob('%s/val/*' % self.args.data_dir), self.args.batch_size)
        sample = [load_data(sample_file, self.args) for sample_file in data]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        sample_images = sample_images[:,:,:,:4]
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        save_images(samples, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.fake_B_sum,
                                       self.real_B_sum,
                                       self.d_loss_fake_sum,
                                       self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.d_loss_real_sum,
                                       self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(20000000):
            data = glob('%s/train/*' % (self.data_dir))
            #np.random.shuffle(data)
            print("start training...")
            batch_idxs = min(len(data), self.args.train_size) // self.batch_size
            print(batch_idxs)
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file, self.args) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_images = batch_images[:,:,:,:4]

                # Update D network only if gan is used
                if self.args.use_gan:
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={ self.real_data: batch_images })
                    self.writer.add_summary(summary_str, counter)
                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={ self.real_data: batch_images })
                    self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                errG = self.g_loss.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errG))

                if np.mod(counter, 100) == 1:
                    self.sample_model(self.args.sample_dir, epoch, idx)

                if np.mod(counter, 500) == 2:
                    self.save(self.args.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        initial_conv_channels = 64
        h0 = lrelu(conv2d(image, initial_conv_channels, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(self.d_bn1(conv2d(h0, initial_conv_channels*2, name='d_h1_conv')))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(self.d_bn2(conv2d(h1, initial_conv_channels*4, name='d_h2_conv')))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(self.d_bn3(conv2d(h2, initial_conv_channels*8, d_h=1, d_w=1, name='d_h3_conv')))
        # h3 is (16 x 16 x self.df_dim*8)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(self, image):
        s = self.output_size
        s2, s4, s8, s16, s32, s64, s128, s256, s512 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128), int(s/256), int(s/512)

        initial_conv_channels = 64
        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, initial_conv_channels, name='g_e1_conv')
        # e1 is (128 x 128 x self.gf_dim)
        e2 = self.g_bn_e2(conv2d(lrelu(e1), initial_conv_channels*2, name='g_e2_conv'))
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = self.g_bn_e3(conv2d(lrelu(e2), initial_conv_channels*4, name='g_e3_conv'))
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = self.g_bn_e4(conv2d(lrelu(e3), initial_conv_channels*8, name='g_e4_conv'))
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = self.g_bn_e5(conv2d(lrelu(e4), initial_conv_channels*8, name='g_e5_conv'))
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = self.g_bn_e6(conv2d(lrelu(e5), initial_conv_channels*8, name='g_e6_conv'))
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = self.g_bn_e7(conv2d(lrelu(e6), initial_conv_channels*8, name='g_e7_conv'))
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = self.g_bn_e8(conv2d(lrelu(e7), initial_conv_channels*8, name='g_e8_conv'))
        # e8 is (1 x 1 x self.gf_dim*8)

        next_dconv_input = e8
        if s == 1024:
            # add additional conv layers
            e9 = self.g_bn_e9(conv2d(lrelu(e8), initial_conv_channels*8, name='g_e9_conv'))
            e10 = self.g_bn_e10(conv2d(lrelu(e9), initial_conv_channels*8, name='g_e10_conv'))
            # add additional deconv layers
            self.d10, self.d10_w, self.d10_b = deconv2d(tf.nn.relu(e10),
                [self.batch_size, s512, s512, initial_conv_channels*8], name='g_d10', with_w=True)
            d10 = tf.nn.dropout(self.g_bn_d9(self.d10), 0.5)
            d10 = tf.concat(3, [d10, e9])
            self.d9, self.d9_w, self.d9_b = deconv2d(tf.nn.relu(d10),
                [self.batch_size, s256, s256, initial_conv_channels*8], name='g_d9', with_w=True)
            d9 = tf.nn.dropout(self.g_bn_d8(self.d9), 0.5)
            d9 = tf.concat(3, [d9, e8])
            next_dconv_input = d9

        self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(next_dconv_input),
            [self.batch_size, s128, s128, initial_conv_channels*8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
        d1 = tf.concat(3, [d1, e7])
        # d1 is (2 x 2 x self.gf_dim*8*2)

        self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
            [self.batch_size, s64, s64, initial_conv_channels*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
        d2 = tf.concat(3, [d2, e6])
        # d2 is (4 x 4 x self.gf_dim*8*2)

        self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
            [self.batch_size, s32, s32, initial_conv_channels*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
        d3 = tf.concat(3, [d3, e5])
        # d3 is (8 x 8 x self.gf_dim*8*2)

        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
            [self.batch_size, s16, s16, initial_conv_channels*8], name='g_d4', with_w=True)
        d4 = self.g_bn_d4(self.d4)
        d4 = tf.concat(3, [d4, e4])
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
            [self.batch_size, s8, s8, initial_conv_channels*4], name='g_d5', with_w=True)
        d5 = self.g_bn_d5(self.d5)
        d5 = tf.concat(3, [d5, e3])
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
            [self.batch_size, s4, s4, initial_conv_channels*2], name='g_d6', with_w=True)
        d6 = self.g_bn_d6(self.d6)
        d6 = tf.concat(3, [d6, e2])
        # d6 is (64 x 64 x self.gf_dim*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
            [self.batch_size, s2, s2, initial_conv_channels], name='g_d7', with_w=True)
        d7 = self.g_bn_d7(self.d7)
        d7 = tf.concat(3, [d7, e1])
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
            [self.args.batch_size, s, s, self.args.output_nc], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d8)

    def sampler(self, image):
        tf.get_variable_scope().reuse_variables()
        return self.generator(image)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.model_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.model_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self):
        """Test pix2pix"""
        tf.initialize_all_variables().run()

        sample_files = glob('%s/test/*' % (self.data_dir))

        # sort testing input
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, self.args, is_test=True) for sample_file in sample_files]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images):
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            save_images(samples, [self.batch_size, 1],
                        './{}/test_{:04d}.png'.format(self.args.test_dir, idx))
