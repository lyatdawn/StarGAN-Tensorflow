# -*- coding:utf-8 -*-
"""
Util module.
"""
import tensorflow as tf
import numpy as np
import cv2

def save_images(noisy, gene_output, label, image_path, max_samples=4):
    # batch_size is 1.
    image = np.concatenate([noisy, gene_output, label], axis=2) # concat 4D array, along width.
    image = image[0:max_samples, :, :, :]
    image = np.concatenate([image[i, :, :, :] for i in range(max_samples)], axis=0)
    # concat 3D array, along axis=0, i.e. along height. shape: (1024, 256, 3/1).

    # save image
    # scipy.misc.toimage(), array is 2D(gray, reshape to (H, W)) or 3D(RGB).
    # scipy.misc.toimage(image, cmin=0., cmax=1.).save(image_path) # image_path contain image path and name.
    # cv.imwrite() save image.
    cv2.imwrite(image_path, np.uint8(image.clip(0., 1.) * 255.))

def res_mod_layers(in_data, num_filters, kernel_size, strides, padding, use_bias, 
        ReflectionPadding, padding_size=None):
    if ReflectionPadding:
        input_data = tf.pad(in_data, padding_size, 'CONSTANT')
    else:
        input_data = in_data
    # conv
    conv_out = tf.layers.conv2d(
        inputs=input_data,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias)
    # use_bias=False/True.
    # Instance Norm
    bn_out = tf.contrib.layers.instance_norm(
        inputs=conv_out,
        center=True,
        scale=True,
        epsilon=1e-05)
    # center represent beta; scale represent gamma. i.e. output = gamma*input + beta.
    # ReLU
    act_out = tf.nn.relu(bn_out)

    return act_out

def res_block(in_data, num_filters, kernel_size, strides, padding, use_bias, 
        ReflectionPadding, padding_size=None):
    # first conv + bn + relu
    res_out = res_mod_layers(in_data=in_data, num_filters=num_filters, kernel_size=kernel_size, strides=strides, 
        padding=padding, use_bias=use_bias, ReflectionPadding=ReflectionPadding, padding_size=padding_size)
    # conv
    resblock_conv_out = tf.layers.conv2d(
        inputs=res_out,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias)
    # Instance Norm
    resblock_bn_out = tf.contrib.layers.instance_norm(
        inputs=resblock_conv_out,
        center=True,
        scale=True,
        epsilon=1e-05)
    # output
    output = in_data + resblock_bn_out

    return output