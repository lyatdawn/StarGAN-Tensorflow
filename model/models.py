# -*- coding:utf-8 -*-
"""
Generator and Discriminator network.
"""
import tensorflow as tf
import numpy as np
import utils

# In style transform, use Instance Norm, instead of Batch Norm.
def network_Gen(name, in_data, c, image_size,  num_filters, image_c, c_dim, n_blocks=6, reuse=False):
    # num_filters=64; image_c=3.
    assert in_data is not None
    with tf.variable_scope(name, reuse=reuse):
        # input_data: the number channel of in_data is image_c + c_dim.
        # c_dim: c_dim is the dimension of condition variable, i.e. the length of selected attribute labels.
        # c_dim is 8.
        # First, should concat input image and attribute label. Refer to StarGAN/model.py. 
        # The channel of input for Generator network is image_c + c_dim.
        # in_data is images shape is [batch_size, H, W, C], c is attribute labels, shape is [batch_size, c_dim].

        # unsqueeze
        c = tf.expand_dims(tf.expand_dims(c, axis=1), axis=2)
        # c shape is [batch_size, c_dim]. Utlize tf.expand_dims() expand the 2nd and 3rd dimension, i.e.
        # shape is [batch_size, 1, 1, c_dim]. 
        # tf.expand_dims(), Inserts a dimension of 1 into a tensor's shape.
        # Finally, c shape is [batch_size, H, W, c_dim]. e.g. H is 128.
        '''
        In pytorch, use c = c.expand(c.size(0), c.size(1), 128, 128) to achieve this goal, In Tensorflow, it can
        utlize tf.tile() to implement this, tf.tile() arguments like:
        input: A Tensor. 1-D or higher.
        multiples: Tensor, e.g.
            input shape is 2 * 1 * 1 * C, multiples is (1, 3, 3, 1), then output shape is 
            (2 * 1, 1 * 3, 1 * 3, C * 1) == (2, 3, 3, C).
        '''
        c = tf.tile(c, (1, image_size, image_size, 1))
        # c shape is [batch_size, H, W, c_dim].

        # int(c.shape[0])
        input_data = tf.concat([in_data, c], axis=3)
        # concat channel dim. concat argument is values: A list of Tensor.
        # input_data shape is [batch_size, H, W, C + c_dim]


        # In conv, if padding='VALID', it will use tf.pad() to pad tensor.
        c_out_res01 = utils.res_mod_layers(in_data=input_data, num_filters=num_filters, kernel_size=7, 
            strides=[1, 1], padding='SAME', use_bias=False, ReflectionPadding=False)
        # kernel_size=7, stride=1, padding=3. i.e. padding='SAME'.
        
        # Down-Sampling. 
        # In Tensorflow, conv2d padding type only have VALID and SAME, the padding is not a number.
        # So, when the feature map size is changing, it could use it will use tf.pad() to pad tensor.
        # then use conv2d. If use tf.pad() to pad tensor, the conv type is "VALID".
        # For example, in Pytorch, kernel_size=4, stride=2, padding=1. padding is padding 0, so in tf.pad()
        # the mode is CONSTANT(constant can change). The padding_size in Tensorflow is corresponding the padding in Pytorch.
        # For 4D tensor, padding is working on the H * W matrix.
        # padding=1, padding_size=[[0,0],[1,1],[1,1],[0,0]].
        # padding=3, padding_size=[[0,0],[3,3],[3,3],[0,0]].
        curr_dim = num_filters
        c_in_G = c_out_res01
        for i in range(2):
            c_out_res02 = utils.res_mod_layers(in_data=c_in_G, num_filters=curr_dim * 2, kernel_size=4, 
                strides=[2, 2], padding="VALID", use_bias=False, ReflectionPadding=True, 
                padding_size=[[0,0],[1,1],[1,1],[0,0]])
            # kernel_size=4, stride=2, padding=1.
            c_in_G = c_out_res02
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(n_blocks):
            c_out_resblock = utils.res_block(in_data=c_in_G, num_filters=curr_dim, 
                kernel_size=3, strides=[1, 1], padding='SAME', use_bias=False, ReflectionPadding=False)
            c_in_G = c_out_resblock

        # Up-Sampling
        for i in range(2):
            '''
            transpose conv is a Up-Sampling operate.
            1) padding is "VALID"
            out = (in - 1) * stride + kernel_size
            2) padding is "SAME"
            out = (in - 1) * stride + kernel_size + stride-1 - 2 * padding
            padding is (kernel_size - 1)/2
            '''
            c_out_deconv = tf.layers.conv2d_transpose(
                inputs=c_in_G,
                filters=int(curr_dim//2),
                kernel_size=4,
                strides=[2, 2],
                padding="SAME",
                use_bias=False)
            # Instance Norm
            c_out_bn = tf.contrib.layers.instance_norm(
                inputs=c_out_deconv,
                center=True,
                scale=True,
                epsilon=1e-05)
            c_out_relu = tf.nn.relu(c_out_bn)
            c_in_G = c_out_relu
            curr_dim = curr_dim // 2

        c_out_conv = tf.layers.conv2d(
            inputs=c_in_G,
            filters=image_c,
            kernel_size=7,
            strides=[1, 1],
            padding="SAME",
            use_bias=False)
        # kernel_size=7, stride=1, padding=3. i.e. padding='SAME'.
        c_out_tanh = tf.nn.tanh(c_out_conv)

    return c_out_tanh
            
def network_Dis(name, in_data, image_size, num_filters, c_dim, n_layers=6, reuse=False):
    # num_filters=64; image_c=3.
    assert in_data is not None
    with tf.variable_scope(name, reuse=reuse):
        # First, use tf.pad() to pad tensor.
        padding_size=[[0,0],[1,1],[1,1],[0,0]]
        input_data = tf.pad(in_data, padding_size, 'CONSTANT')
        c_out_conv_1 = tf.layers.conv2d(
            inputs=input_data,
            filters=num_filters,
            kernel_size=4,
            strides=[2, 2],
            padding="VALID")
        c_out_relu_1 = tf.nn.leaky_relu(features=c_out_conv_1, alpha=0.01)

        curr_dim = num_filters
        c_in_D = c_out_relu_1
        for i in range(1, n_layers):
            # First, use tf.pad() to pad tensor.
            padding_size=[[0,0],[1,1],[1,1],[0,0]]
            c_in_D = tf.pad(c_in_D, padding_size, 'CONSTANT')
            c_out_conv_2 = tf.layers.conv2d(
                inputs=c_in_D,
                filters=curr_dim * 2,
                kernel_size=4,
                strides=[2, 2],
                padding="VALID")
            c_out_relu_2 = tf.nn.leaky_relu(features=c_out_conv_2, alpha=0.01)
            c_in_D = c_out_relu_2
            curr_dim = curr_dim * 2

        c_out_conv_3 = tf.layers.conv2d(
            inputs=c_in_D,
            filters=1,
            kernel_size=3,
            strides=[1, 1],
            padding="SAME",
            use_bias=False)

        k_size = int(image_size / np.power(2, n_layers))
        c_out_conv_4 = tf.layers.conv2d(
            inputs=c_in_D,
            filters=c_dim,
            kernel_size=k_size,
            strides=[1, 1],
            padding="VALID",
            use_bias=False)
        # kernel_size=k_size, stride=1, padding=0. i.e. VALID.

    # squeeze
    # tf.squeeze(), Removes the dimensions of size 1. 
    # axis Defaults to [], Default to remove all the dimensions of size 1.
    return tf.squeeze(c_out_conv_3), tf.squeeze(c_out_conv_4)