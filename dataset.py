import os
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class ImageDataset:
    def __init__(self, **params):
        # Parse args
        self.data_dir = params['data_dir']
        self.batch_size = params['batch_size']
        self.transform_flags = params['transform_flags']

        # Create paths
        self.train_path = os.path.join(self.data_dir, 'training')
        self.val_path = os.path.join(self.data_dir, 'validation')
        self.test_path = os.path.join(self.data_dir, 'test')

        # Create data-sets
        self.train_ds = self.create_dataset(self.train_path)
        self.val_ds = self.create_dataset(self.val_path)
        self.test_ds = self.create_dataset(self.test_path)

    def create_dataset(self, data_dir):
        list_ds = tf.data.Dataset.list_files(data_dir + '/*.jpg')
        images_ds = list_ds.map(self.parse_image_file)

        transform_list = [
            self.random_brightness,
            self.random_contrast,
            self.random_flip,
            self.random_rotate_img
        ]
        for idx, flag in enumerate(self.transform_flags):
            if flag:
                images_ds = images_ds.map(transform_list[idx])

        images_ds = images_ds.batch(self.batch_size)

        return images_ds

    @staticmethod
    def parse_image_file(file_path):
        file_name = tf.strings.split(file_path, '/')[-1]
        score_str = tf.strings.split(file_name, '_')[0]
        score = tf.strings.to_number(score_str)

        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image, score

    @staticmethod
    def random_rotate_img(image, score):
        image_shape = image.shape

        def rotate_image(im):
            angle = np.random.uniform(-30, 30)
            im = scipy.ndimage.rotate(im, angle, reshape=False)
            return im

        # Wraps a python function into a TensorFlow op that executes it eagerly.
        [image, ] = tf.py_function(rotate_image, [image], [tf.float32])
        image.set_shape(image_shape)
        return image, score

    @staticmethod
    def random_flip(image, score):
        image = tf.image.flip_left_right(image)
        return image, score

    @staticmethod
    def random_contrast(image, score):
        image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
        return image, score

    @staticmethod
    def random_brightness(image, score):
        image = tf.image.random_brightness(image, 0.1)
        return image, score


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(str(label.numpy()))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    dataset_obj = ImageDataset(data_dir='data',
                               batch_size=4,
                               transform_flags=[0, 0, 1, 0])

    images_ds = dataset_obj.train_ds

    for img, scr in images_ds:
        print(img.shape, scr.shape)
