from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib
from presnet_train import train
from PIL import Image
from presnet import *
import tensorflow as tf
import numpy as np
IMAGE_SIZE=256
NUM_CLASSES=48
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
tf.app.flags.DEFINE_string('logs_dir', 'H:\\tmp\\resnet_train', 'path')

image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")


# pred_annotation, logits = inference(image, keep_probability)


logits = inference(image,
                   num_classes=NUM_CLASSES,
                   is_training=False,
                   use_bias=(not FLAGS.use_bn),
                   num_blocks=[3, 4, 6, 3])

imgage = Image.open("../0a55be2bb250ca8c037f60dd9a7c5997.jpg")
img = imgage.resize((IMAGE_SIZE, IMAGE_SIZE))
sess = tf.Session()

print("Setting up Saver...")
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...")
img=np.reshape(img,[-1,IMAGE_SIZE,IMAGE_SIZE,3])
pred = sess.run(logits, feed_dict={image: img})
result=pred.astype(int)
print(result)

labelposion=result[0]

for pos in range(48):
    if pos % 2 == 0:
        labelposion[pos] = int(labelposion[pos] / IMAGE_SIZE * imgage.width)
    else:
        labelposion[pos] = int(labelposion[pos] / IMAGE_SIZE * imgage.height)

print(labelposion)



