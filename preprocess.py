import numpy as np 
import tensorflow as tf
import pandas as pd
import os



'''
Loads and converts an image to float32
'''
def clean_image(file_path):
    # Load & normalize image
    print("BEFORE DECODE")
    image = tf.io.decode_png(tf.io.read_file(file_path), channels=3)
    print("AFTER DECODE")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256,256])
    image = (image - 0.5) * 2 # Rescale data to range (-1, 1) #TODO do we need this?
    print(image)
    return image


'''
    Returns a tuple of the image paths and corresponding label paths. 
'''
def get_data_paths(dir_path):

    img_paths, label_paths = [], []
    for f_name in os.listdir(dir_path + '/images'):
        img_path = dir_path + '/images/' + f_name
        label_path = dir_path + '/labels/' + f_name
        img_paths.append(img_path)
        label_paths.append(label_path)
    
    return (img_paths, label_paths)


'''
Loads the image and label maks. 
'''
def get_image_label_pair(img_path, label_path):
  img = clean_image(img_path)
  #label = tf.io.read_file(label_path)
  label = clean_image(label_path)
  # TODO maybe clean label image ??

  return label, img


'''
Builds a dataset of images from the given directory.
'''
def load_images(train_data_dir, batch_size=1):
    print("LOADING IMAGES FROM: ", train_data_dir)

    data_paths = get_data_paths(train_data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(data_paths)
    #dataset = dataset.shuffle(buffer_size=250000)
    dataset = dataset.map(map_func=get_image_label_pair, num_parallel_calls=2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


# Example usage
# dataset = load_images("data/train")
# for imgs, labels in dataset.as_numpy_iterator():
#     # imgs is a list of batch_size of images
#     # labels is a list of batch_size of labels
#     # labels[0] is the label mask for imgs[0]
#     print("num imgs: ", len(imgs))
#     print("num labels: ", len(labels))
    

