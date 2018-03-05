# -*- coding:utf-8 -*-
"""
An implementation of CycleGan using TensorFlow (work in progress).
"""
import os
import glob
import tensorflow as tf
import numpy as np
from model import stargan
import cv2
import scipy.misc # save image


def main(_):
    tf_flags = tf.app.flags.FLAGS
    # gpu config.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    if tf_flags.phase == "train":
        with tf.Session(config=config) as sess: 
        # when use queue to load data, not use with to define sess
            train_model = stargan.StarGAN(sess, tf_flags)
            train_model.train(tf_flags.image_root, tf_flags.metadata_path, tf_flags.training_steps, 
                              tf_flags.summary_steps, tf_flags.checkpoint_steps, tf_flags.save_steps)
    else:
        with tf.Session(config=config) as sess:
            # test on a image pair.
            test_model = stargan.StarGAN(sess, tf_flags)
            test_model.load(tf_flags.checkpoint)
            # image path
            image_path = glob.glob(os.path.join("./datasets", "test", "*.jpg"))
            generated_images = test_model.test(image_path=image_path)
            # return numpy ndarray.
            print(generated_images)
            
            # save images.
            save_images = np.concatenate([generated_images[i][0, :] for i in range(len(generated_images))], 
                axis=1)
            
            # scipy.misc.toimage() has problem.
            # image_path contain image path and name.
            cv2.imwrite("results.jpg", np.uint8(save_images.clip(0., 1.) * 255.))

            # Utilize cv2.imwrite() to save images.
            print("Saved testing files.")

if __name__ == '__main__':
    tf.app.flags.DEFINE_string("output_dir", "model_output", 
                               "checkpoint and summary directory.")
    tf.app.flags.DEFINE_string("phase", "train", 
                               "model phase: train/test.")
    tf.app.flags.DEFINE_string("image_root", "./datasets/CelebA_nocrop/images", 
                               "the root path of images.")
    tf.app.flags.DEFINE_string("metadata_path", "./datasets/list_attr_celeba.txt", 
                               "the path of metadata.")
    tf.app.flags.DEFINE_integer("batch_size", 64, 
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("c_dim", 8, 
                                "the dimension of condition.")
    tf.app.flags.DEFINE_float("lambda_cls", 1., 
                              "scale cls loss.")
    tf.app.flags.DEFINE_float("lambda_rec", 10., 
                              "scale G rec loss.")
    tf.app.flags.DEFINE_float("lambda_gp", 10., 
                              "scale gradient penalty loss.")
    tf.app.flags.DEFINE_integer("d_train_repeat", 5, 
                                "the frequency of training Discriminator network.")
    tf.app.flags.DEFINE_integer("training_steps", 100000, 
                                "total training steps.")
    tf.app.flags.DEFINE_integer("summary_steps", 100, 
                                "summary period.")
    tf.app.flags.DEFINE_integer("checkpoint_steps", 1000, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_integer("save_steps", 500, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_string("checkpoint", None, 
                                "checkpoint name for restoring.")
    tf.app.run(main=main)
