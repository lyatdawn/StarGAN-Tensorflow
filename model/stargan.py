# -*- coding:utf-8 -*-
"""
Load data.
Train and test StarGAN.
"""
import os
import logging
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import cv2
from models import network_Gen, network_Dis
from utils import save_images
import sys
sys.path.append("../data")
from data import data_loader

class StarGAN(object):
    def __init__(self, sess, tf_flags):
        self.sess = sess
        self.dtype = tf.float32

        # checkpoint and summary.
        self.output_dir = tf_flags.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        self.checkpoint_prefix = "model"
        self.saver_name = "checkpoint"
        self.summary_dir = os.path.join(self.output_dir, "summary")

        self.is_training = (tf_flags.phase == "train") # train or test.
        self.d_train_repeat = tf_flags.d_train_repeat
        self.sample_dir = "sample"

        # placeholder, clean_images and noisy_images.
        self.batch_size = tf_flags.batch_size
        self.image_h = 128
        self.image_w = 128
        assert self.image_h == self.image_w
        self.image_c = 3
        self.c_dim = tf_flags.c_dim

        # placeholder. images and attribute labels.
        # real_images.
        # attribute labels contain a real_c, a fake_c. c is condition, the dimension of condition is c_dim, 
        # default is 8. Here is noly define placeholder, the true data is at the process of training.
        self.real_images = tf.placeholder(self.dtype, [None, self.image_h, self.image_w, self.image_c])
        self.real_c = tf.placeholder(self.dtype, [None, self.c_dim])
        self.fake_c = tf.placeholder(self.dtype, [None, self.c_dim])
        # TODO: the learning rate could also be a placeholder, so it can be change in the training.
        # alpha placeholder. Utlize alpha to sample a pair of a real and a generated images.
        # In the process of training, must change! so define alpha is placeholder.
        self.alpha = tf.placeholder(self.dtype, [None, self.image_h, self.image_w, self.image_c])

        # train
        if self.is_training:
            # parameters
            self.lambda_cls = tf_flags.lambda_cls
            self.lambda_rec = tf_flags.lambda_rec
            self.lambda_gp = tf_flags.lambda_gp

            # makedir aux dir
            self._make_aux_dirs()
            # compute and define loss
            self._build_training()
            # logging, only use in training
            log_file = os.path.join(self.output_dir, "StarGAN.log")
            logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                                filename=log_file,
                                level=logging.DEBUG,
                                filemode='a+')
            logging.getLogger().addHandler(logging.StreamHandler())
        else:
            # test
            self._build_test()

    def _build_training(self):
        # Generator network. Refer to CycleGAN, there will use Generator network for 2 times.
        # one output the target of task, the other want to approximate the real image.
        # Since the target of task is not exist, so is the unsupervised task. And, this task if one-to-multi, 
        # so we can use Conditional GAN rather than Standard GAN.
        # input real images and fake_c, output the fake images. fake images is our target, e.g. Aged image.
        self.fake_images = network_Gen(name="G", in_data=self.real_images, c=self.fake_c, image_size=self.image_h,
            num_filters=64, image_c=self.image_c, c_dim=self.c_dim, reuse=False)
        # input fake_images and real_c, output rec_images. i.e. reconstruction images. rec_images want to 
        # approximate the real image.
        self.rec_images = network_Gen(name="G", in_data=self.fake_images, c=self.real_c, image_size=self.image_h,
            num_filters=64, image_c=self.image_c, c_dim=self.c_dim, reuse=True)
        # Generator network Variables.
        self.G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")

        # Discriminator network. Refer to StarGAN, since there are not target images, the task is unsupervised task,
        # so the D network input need to change a little. 
        # The input of D is no longer target and fake_images, but the real_images and fake_images. output, one is 
        # pre_real(predict real), the other is pre_fake(predict fake).
        # Discriminator network has two output, one is for images, the other is for attribute label. 
        # i.e. src and cls.
        self.pre_fake_src, self.pre_fake_cls = network_Dis(name="D", in_data=self.fake_images, 
            image_size=self.image_h, num_filters=64, c_dim=self.c_dim, reuse=False)

        self.pre_real_src, self.pre_real_cls = network_Dis(name="D", in_data=self.real_images, 
            image_size=self.image_h, num_filters=64, c_dim=self.c_dim, reuse=True)

        # Compute gradient penalty. In Pytorch code, use torch.autograd.grad computes and returns the gradient of
        # outputs w.r.t. the inputs. In Tensorflow, it can use tf.gradients() to implement this.
        # First, define x^, x^ is sampled uniformly along a straight line between a pair of a real and .
        # a generated images.
        # alpha, a uniform distribution tensor, shape is [batch_size, h, w, c]. define alpha is a placeholder.
        # create a Variable. define directly!
        self.interpolated = self.alpha * self.real_images + (1 - self.alpha) * self.fake_images
        self.pre_inter_src, _ = network_Dis(name="D", in_data=self.interpolated, 
            image_size=self.image_h, num_filters=64, c_dim=self.c_dim, reuse=True)
        
        # Discriminator network Variables.
        self.D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

        # loss
        # In paper, To stabilize the training process and generate higher quality images, utlize WGAN.
        # In the code, the Discriminator network loss is implemented by torch.mean(out_src).
        # Since the paper use WGAN to implement StarGAN, the D real loss and D fake loss is torch.mean(out_src).
        # See the equation (8) in paper for detial.
        # In this implement, I write three type loss: 1. WGAN loss; 2. Standard GAN loss; 3. lsgan loss.
        # In this, temporarily only use lsgan to define the G and D loss. See equation (1) for detial.
        # lsgan might not have effect, so it can try WGAN to define loss.

        # 1. WGAN loss.
        # Discriminator loss
        # D real loss.
        self.D_real_loss = -tf.reduce_mean(self.pre_real_src)
        # D fake loss, input is fake images(target).
        self.D_fake_loss = tf.reduce_mean(self.pre_fake_src)
        # D cls loss
        # in Pytorch code, the d cls loss use 
        # F.binary_cross_entropy_with_logits(out_cls, real_label, size_average=False). argument is input, target.
        # In Tensorflow, it can use tf.nn.sigmoid_cross_entropy_with_logits() to implement this.
        # tf.reduce_mean(tf.squared_difference()), don't forget tf.reduce_mean()!!
        self.D_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.real_c, logits=self.pre_real_cls))
        
        # Compute gradient penalty. In Pytorch code, use torch.autograd.grad computes and returns the gradient of
        # outputs w.r.t. the inputs. In Tensorflow, it can use tf.gradients() to implement this.
        grad = tf.gradients(ys=self.pre_inter_src, xs=[self.interpolated])[0]
        # compute and return the gradient of ys/outputs w.r.t. the xs/inputs.
        # xs = [].
        # return A list of sum(dys/dx) for each x in xs.
        # gradient penalty.
        grad_l2norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=3))
        # grad may not use reshape.
        self.D_loss_gp = tf.reduce_mean(
            tf.squared_difference(grad_l2norm, tf.ones_like(grad_l2norm)))

        self.D_loss = self.D_real_loss + self.D_fake_loss + self.lambda_cls * self.D_cls_loss + \
            self.lambda_gp * self.D_loss_gp
        # Compute gradient penalty has problem, result to Nan!!
        # is tf.gradients(ys=self.pre_inter_src, xs=[self.interpolated])[0], not
        # tf.gradients(ys=self.pre_inter_src[0], xs=[self.interpolated])[0].

        # Generator loss
        # generator loss
        self.G_gen_loss = -tf.reduce_mean(self.pre_fake_src)
        # G rec loss, input is fake images, output is rec iamges.
        self.G_rec_loss = tf.reduce_mean(tf.abs(self.rec_images - self.real_images))
        # G cls loss
        # in Pytorch code, the d cls loss use 
        # F.binary_cross_entropy_with_logits(out_cls, real_label, size_average=False). argument is input, target.
        # In Tensorflow, it can use tf.nn.sigmoid_cross_entropy_with_logits() to implement this.
        # # tf.reduce_mean(tf.squared_difference()), don't forget tf.reduce_mean()!!
        self.G_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.fake_c, logits=self.pre_fake_cls))

        # G loss
        self.G_loss = self.G_gen_loss + self.lambda_rec * self.G_rec_loss + self.lambda_cls * self.G_cls_loss

        # 2. Standard GAN loss is log loss, In pytorch, utlize Binary Cross Entropy loss. In tensorflow, utlize 
        # Sigmoid Cross Entropy loss. 
        # In loss_helper.py, replace tf.reduce_mean(tf.squared_difference()) to 
        # tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits()) is Standard GAN loss.

        # 3. lsgan to define the G and D loss., the code can see loss_helper.py.
        
        # optimizer
        # appoint the variable list.
        # In Pytorch code, the learning rate of G and D is changing, e.g.:
        # lr = lr - lr / lr_factor. Refer to srez code to update lr, lr could be a placeholder, so it can update
        # in the process of training.
        self.G_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999).minimize(
            self.G_loss, var_list=self.G_variables, name="G_opt")
        self.D_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.999).minimize(
            self.D_loss, var_list=self.D_variables, name="D_opt")

        # summary
        tf.summary.scalar('G_gen_loss', self.G_gen_loss)
        tf.summary.scalar('G_rec_loss', self.G_rec_loss)
        tf.summary.scalar('G_cls_loss', self.G_cls_loss)
        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('D_real_loss', self.D_real_loss)
        tf.summary.scalar('D_fake_loss', self.D_fake_loss)
        tf.summary.scalar('D_cls_loss', self.D_cls_loss)
        tf.summary.scalar('D_loss_gp', self.D_loss_gp)
        tf.summary.scalar('D_loss', self.D_loss)

        self.summary = tf.summary.merge_all()
        # summary and checkpoint
        self.writer = tf.summary.FileWriter(
            self.summary_dir, graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()

    def train(self, image_root, metadata_path, training_steps, summary_steps, checkpoint_steps, save_steps):
        step_num = 0
        # restore last checkpoint
        latest_checkpoint = tf.train.latest_checkpoint("model_output_20180303172402/checkpoint") 
        # use pretrained model, it can be self.checkpoint_dir, "", or you can appoint the saved checkpoint path.
        # e.g., model_output_20180303114343/checkpoint
        
        if latest_checkpoint:
            step_num = int(os.path.basename(latest_checkpoint).split("-")[1])
            assert step_num > 0, "Please ensure checkpoint format is model-*.*."
            self.saver.restore(self.sess, latest_checkpoint)
            logging.info("{}: Resume training from step {}. Loaded checkpoint {}".format(datetime.now(), 
                step_num, latest_checkpoint))
        else:
            self.sess.run(tf.global_variables_initializer()) # init all variables
            logging.info("{}: Init new training".format(datetime.now()))

        # data
        # define a object of CelebADataset.
        dataset = data_loader.CelebADataset(image_root=image_root, metadata_path=metadata_path, 
            is_training=self.is_training, batch_size=self.batch_size, image_h=self.image_h, image_w=self.image_w,
            image_c=self.image_c)
        data_generate = dataset.batch_generator_numpy()
        # print(type(data_generate))

        # train
        c_time = time.time()
        for c_step in xrange(step_num + 1, training_steps + 1):
            # alpha, a uniform distribution tensor, shape is [batch_size, h, w, c].
            # In the process of training, must change! so define alpha is placeholder. Then appoint alpha.
            # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
            alpha = np.random.rand(self.batch_size, self.image_h, self.image_w, self.image_c)
            # minval is 0, maxval is 1. There, must be numpy ndarray.

            data_gen = data_generate.next()
            # name must be different!!
            # data_gen["images"] and data_gen["attribute"]. data_gen["attribute"] is the true data for real_c.
            # generate fake_c. Utlize np.random.shuffle() to shuffle the numpy array, refer to CycleGAN-Tensorflow,
            # utlize numpy load data. First define a index, then shuffle this index, finally shuffle the array.
            rand_index = np.arange(self.batch_size) # length is batch_size.
            # not fixed order
            np.random.shuffle(rand_index)
            fake_c = data_gen["attribute"][rand_index]

            c_feed_dict = {
                # numpy ndarray
                self.real_images: data_gen["images"],
                self.real_c: data_gen["attribute"],
                self.fake_c: fake_c,
                self.alpha: alpha
                # TODO: the learning rate could also be a placeholder, so it can be change in the training.
            }

            # Refer to Pytorch StarGAN, train D network d_train_repeat times, then train G network one time.
            '''
            # G and D are training at the same time.
            self.ops = [self.G_opt, self.D_opt]
            self.sess.run(self.ops, feed_dict=c_feed_dict)
            '''
            # train D network d_train_repeat times, then train G network one time.
            # train D.
            self.sess.run(self.D_opt, feed_dict=c_feed_dict)
            # train G
            if c_step % self.d_train_repeat == 0:
                self.sess.run(self.G_opt, feed_dict=c_feed_dict)

            # save summary
            if c_step % summary_steps == 0:
                c_summary = self.sess.run(self.summary, feed_dict=c_feed_dict)
                self.writer.add_summary(c_summary, c_step)

                e_time = time.time() - c_time
                time_periter = e_time / summary_steps
                logging.info("{}: Iteration_{} ({:.4f}s/iter) {}".format(
                    datetime.now(), c_step, time_periter,
                    self._print_summary(c_summary)))
                c_time = time.time() # update time

            # save checkpoint
            if c_step % checkpoint_steps == 0:
                self.saver.save(self.sess,
                    os.path.join(self.checkpoint_dir, self.checkpoint_prefix),
                    global_step=c_step)
                logging.info("{}: Iteration_{} Saved checkpoint".format(
                    datetime.now(), c_step))

            # save training images
            if c_step % save_steps == 0:
                # In here, save training images. In the saved images, every line contains 3 images. i.e.
                # real_images, fake_images, rec_images.
                real_images, fake_images, rec_images= self.sess.run(
                    [self.real_images, self.fake_images, self.rec_images],
                    feed_dict=c_feed_dict)
                save_images(real_images, fake_images, rec_images,
                    './{}/train_{}_{:06d}.png'.format(self.sample_dir, "stargan", c_step))

        logging.info("{}: Done training".format(datetime.now()))

    def _build_test(self):
        # Generator network. network_Gen() is Generator network.
        # input real images and fake_c, output the fake images. fake images is our target, e.g. Aged image.
        self.fake_images = network_Gen(name="G", in_data=self.real_images, c=self.fake_c, image_size=self.image_h,
            num_filters=64, image_c=self.image_c, c_dim=self.c_dim, reuse=False)

        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name) 
        # define saver, after the network!

    def load(self, checkpoint_name=None):
        # restore checkpoint
        print("{}: Loading checkpoint...".format(datetime.now())),
        if checkpoint_name:
            checkpoint = os.path.join(self.checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, checkpoint)
            print(" loaded {}".format(checkpoint_name))
        else:
            # restore latest model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.checkpoint_dir)
            if latest_checkpoint:
                self.saver.restore(self.sess, latest_checkpoint)
                print(" loaded {}".format(os.path.basename(latest_checkpoint)))
            else:
                raise IOError(
                    "No checkpoints found in {}".format(self.checkpoint_dir))

    def test(self, image_path=None):
        # In the phase, 1) you can use the 2000 images(see data/data_loader.py for detial, 
        # the number of test images is 2000) to generate one mode's generated images. e.g.:
        # image --> G --> target. In this way, the process of testing is similar with training, you can refer to 
        # the process of "data" and "save training images" to obtain one mode's generated images.
        # 2) use a small image dataset, e.g. only one images. input a real images, generate all the mode's images.
        # i.e. generate the 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Male', 'Smiling', 'Wearing_Hat', 
        # 'Young' images. the fake_c is a ont-hot vector.
        # 2) can refer to Pytorch code, solver.py.
        
        # There, I will use 2) to test the trained model.
        if image_path is not None:
            gen_images = []
            # In tensorflow, test image must divide 255.0.
            image = np.reshape(cv2.resize(cv2.imread(str(image_path[0])), 
                (self.image_h, self.image_w)), (1, self.image_h, self.image_w, self.image_c)) / 255.
            gen_images.append(image)

            # self.c_dim, the dim of attribute label. Default is 8.
            # Generate c_dim * c_dim Zero matrix. Each line is a one-hot vector.
            # Can use tf.one_hot() to get a ont-hot matrix.
            # 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Male', 'Smiling', 'Wearing_Hat', 
            # 'Young' arrtibute.
            indices = [0,1,2,3,4,5,6,7]
            label = self.sess.run(tf.one_hot(indices, depth=8))
            # label[0] = np.array([1,1,1,1,1,1,1,1])
            # print(label)

            for i in range(self.c_dim):
                print("Generate {}th mode's image").format(i)
                c_feed_dict = {
                    # numpy ndarray
                    self.real_images: image,
                    self.fake_c: np.reshape(label[i], (1, self.c_dim))
                }

                gen_image = self.sess.run(self.fake_images, feed_dict=c_feed_dict)
                gen_images.append(gen_image)

        return gen_images

    def _make_aux_dirs(self):
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def _print_summary(self, summary_string):
        self.summary_proto.ParseFromString(summary_string)
        result = []
        for val in self.summary_proto.value:
            result.append("({}={})".format(val.tag, val.simple_value))
        return " ".join(result)