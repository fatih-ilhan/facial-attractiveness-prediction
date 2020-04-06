import os
import tensorflow as tf
from config import Config

from models.cnn import *
from dataset import *


model_name = 'cnn'
data_folder = 'data'
training_data_folder = os.path.join(data_folder, "training")
validation_data_folder = os.path.join(data_folder, "validation")
testing_data_folder = os.path.join(data_folder, "test")

training_dataset = tf.data.Dataset.list_files(training_data_folder + '/*.jpg')
validation_dataset = tf.data.Dataset.list_files(validation_data_folder + '/*.jpg')
testing_dataset = tf.data.Dataset.list_files(testing_data_folder + '/*.jpg')

config_obj = Config(model_name)

optimizer_dispatcher = {"adam": tf.optimizers.Adam}
initializer_dispatcher = {"glorot": tf.initializers.glorot_uniform}

print("Starting experiments")

for conf in config_obj.conf_list:
    batch_size = conf["batch_size"]
    learning_rate = conf["learning_rate"]
    num_epochs = conf["num_epochs"]
    optimizer_type = conf["optimizer_type"]
    initializer_type = conf["initializer_type"]

    training_dataset_ = training_dataset.shuffle(1024).batch(batch_size)

    optimizer = optimizer_dispatcher[optimizer_type](learning_rate)
    initializer = initializer_dispatcher[initializer_type]()

    weights = get_weights(initializer)

    def train_step(model, inputs, outputs):
        with tf.GradientTape() as tape:
            current_loss = loss(model(inputs, weights), outputs)
        grads = tape.gradient(current_loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        print(tf.reduce_mean(current_loss))
        return current_loss.numpy()

    for e in range(num_epochs):
        for file_path_tensor in training_dataset_:
            images = []
            labels = []
            for i in range(file_path_tensor.shape[0]):
                image, label = process_path(file_path_tensor[i])
                images.append(image)
                labels.append(label)
            loss_array = train_step(model, tf.stack(images), tf.convert_to_tensor(labels))