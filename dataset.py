import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_image_info(file_path):
    file_name = tf.strings.split(file_path, os.path.sep)[-1].numpy()
    file_name = str(file_name)[2: -1]

    score = int(file_name[0])
    race = file_name[2]
    gender = file_name[3]
    id = file_name[4:]

    image_info = {"score": score,
                  "race": race,
                  "gender": gender,
                  "id": id}

    return image_info


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def process_path(file_path):
    label = get_image_info(file_path)["score"]
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    return image, label


if __name__ == '__main__':
    data_dir = 'data/training'
    list_ds = tf.data.Dataset.list_files(data_dir + '/*.jpg')

    for file_path in list_ds.take(1):
        image, label = process_path(file_path)
        print("Image shape: ", image.numpy().shape)
        print("Image label: ", label)