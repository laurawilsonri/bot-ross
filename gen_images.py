from main import load_images
import tensorflow as tf
from matplotlib import pyplot as plt


def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  # show example
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


def generate_from_checkpoint():
    BATCH_SIZE = 5

    checkpoint_dir = 'training_checkpoints'
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir) #latest_filename=None
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    test_dataset = load_images("data/train", batch_size=BATCH_SIZE)

    for inp, tar in test_dataset.take(5):
        generate_images(generator, inp, tar)


generate_from_checkpoint()