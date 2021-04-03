import numpy as np 
import tensorflow as tf
import pandas as pd
import os

'''
Loads and converts an image to float32
'''
def clean_image(file_path):
    # Load & normalize image
    image = tf.io.decode_png(tf.io.read_file(file_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - 0.5) * 2 # Rescale data to range (-1, 1)
    return image


'''
Builds a dataset of images from the given directory
'''
def load_images(train_data_dir):
    dir_path = train_data_dir + '/*.png'
    print("LOADING IMAGES FROM: ", dir_path)
    
    dataset = tf.data.Dataset.list_files(dir_path)
    dataset = dataset.shuffle(buffer_size=250000)
    dataset = dataset.map(map_func=clean_image, num_parallel_calls=2)
    dataset = dataset.batch(100, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


# Example usage
images = load_images("data/train/images")
for image in images:
    print(image)

