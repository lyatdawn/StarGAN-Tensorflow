# 2. GAN to define the G and D loss. Refer to the paper, the G loss and D loss can see 
# equation (5) and (6). 
# Discriminator loss
# D real loss, input is real images.
self.D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(self.pre_real_src), logits=self.pre_real_src))
# D fake loss, input is fake images(target).
self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(self.pre_fake_src), logits=self.pre_fake_src))
# D cls loss, in Pytorch code, the d cls loss use 
# F.binary_cross_entropy_with_logits(out_cls, real_label, size_average=False). argument is input, target.
# In Tensorflow, it can use tf.nn.sigmoid_cross_entropy_with_logits() to implement this. argument is labels, logits.
# # tf.reduce_mean(tf.squared_difference()), don't forget tf.reduce_mean()!!
# Only use in real images.
self.D_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=self.real_c, logits=self.pre_real_cls))
self.D_loss = self.D_real_loss + self.D_fake_loss + self.lambda_cls * self.D_cls_loss

# Generator loss
# generator loss
self.G_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(self.pre_fake_src), logits=self.pre_fake_src))
# G rec loss, input is fake images, output is rec iamges.
self.G_rec_loss = tf.reduce_mean(tf.abs(self.rec_images - self.real_images))
# G cls loss
# binary cross entropy with logits, In Tensorflow, it can use tf.nn.sigmoid_cross_entropy_with_logits() to implement this
self.G_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=self.fake_c, logits=self.pre_fake_cls))
# G loss
self.G_loss = self.G_gen_loss + self.lambda_rec * self.G_rec_loss + self.lambda_cls * self.G_cls_loss


# 3. lsgan to define the G and D loss. Refer to the paper, the G loss and D loss can see 
# equation (5) and (6). 
# Discriminator loss
# D real loss, input is real images.
self.D_real_loss = tf.reduce_mean(
    tf.squared_difference(self.pre_real_src, tf.ones_like(self.pre_real_src)))
# D fake loss, input is fake images(target).
self.D_fake_loss = tf.reduce_mean(
    tf.squared_difference(self.pre_fake_src, tf.zeros_like(self.pre_fake_src)))
# D cls loss, in Pytorch code, the d cls loss use 
# F.binary_cross_entropy_with_logits(out_cls, real_label, size_average=False). argument is input, target.
# In Tensorflow, it can use tf.nn.sigmoid_cross_entropy_with_logits() to implement this. argument is labels, logits.
# # tf.reduce_mean(tf.squared_difference()), don't forget tf.reduce_mean()!!
# Only use in real images.
self.D_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=self.real_c, logits=self.pre_real_cls))
self.D_loss = self.D_real_loss + self.D_fake_loss + self.lambda_cls * self.D_cls_loss

# Generator loss
# generator loss
self.G_gen_loss = tf.reduce_mean(
    tf.squared_difference(self.pre_fake_src, tf.ones_like(self.pre_fake_src)))
# G rec loss, input is fake images, output is rec iamges.
self.G_rec_loss = tf.reduce_mean(tf.abs(self.rec_images - self.real_images))
# G cls loss
# binary cross entropy with logits, In Tensorflow, it can use tf.nn.sigmoid_cross_entropy_with_logits() to implement this
self.G_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=self.fake_c, logits=self.pre_fake_cls))
# G loss
self.G_loss = self.G_gen_loss + self.lambda_rec * self.G_rec_loss + self.lambda_cls * self.G_cls_loss