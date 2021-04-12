import tensorflow as tf
import os
import time

from matplotlib import pyplot as plt
#from IPython import display

def load(image_name):
  image = tf.io.read_file('data/train/images/' + image_name)
  image = tf.image.decode_png(image)

  label = tf.io.read_file('data/train/labels/' + image_name)
  label = tf.image.decode_png(label)

  image = tf.cast(image, tf.float32)
  label = tf.cast(label, tf.float32)

  return image, label


PATH = 'data/train/'
re, inp = load('painting6.png')

# casting to int for matplotlib to show the image
# TO SHOW AN EXAMPLE!!
# plt.figure()
# plt.imshow(inp/255.0)
# plt.figure()
# plt.imshow(re/255.0)

# plt.show()