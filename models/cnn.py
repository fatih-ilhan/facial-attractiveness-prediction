import time
from copy import deepcopy

import tensorflow as tf


class CNN:
    def __init__(self, **params):
        # finetune params
        self.initializer = params['initializer']
        self.optim = params['optimizer']
        self.loss_fun = params['loss_function']
        self.alpha = params['alpha']
        self.batch_reg = params['batch_reg']

        self.lmd = params['lambda']
        self.dropout_rate = params['dropout_rate']

        # set filter, pool and dense shapes
        self.filter_shapes = [[5, 5, 3, 16],
                              [5, 5, 16, 16],
                              [3, 3, 16, 32],
                              [3, 3, 32, 32]]
        self.dense_shapes = [[1152, 256],
                             [256, 1]]
        self.pool_shapes = [5, 5, 3, 3]
        self.cnn_strides = [3, 1, 1, 1]
        self.pool_strides = [1, 1, 1, 1]
        self.weight_shapes = self.filter_shapes + self.dense_shapes

        # arrange batch-norm params
        # initialize weights according to initializer
        self.weights = self.init_weights(self.weight_shapes, 'weight')

        if self.batch_reg:
            beta_list = []
            gamma_list = []
            for i in range(len(self.filter_shapes)):
                beta = tf.constant(0.0, shape=[self.filter_shapes[i][-1]])
                beta_list.append(tf.Variable(initial_value=beta,
                                 name='{}_{}'.format('beta', i),
                                 trainable=True,
                                 dtype=tf.float32))
                gamma = tf.constant(1.0, shape=[self.filter_shapes[i][-1]])
                gamma_list.append(tf.Variable(initial_value=gamma,
                                  name='{}_{}'.format('gamma', i),
                                  trainable=True,
                                  dtype=tf.float32))
            self.epsilon = 0.0001
            self.weights += beta_list
            self.weights += gamma_list

    def fit(self, train_ds, eval_ds, num_epochs, early_stop_tolerance):
        """
        :param train_ds: :obj:tf.Dataset
        :param eval_ds: :obj:tf.Dataset
        :param num_epochs: int
        :return: train_loss, eval_loss
        """
        print('Training starts...')
        train_loss = []
        eval_loss = []

        tolerance = 0
        best_eval_loss = 1e6
        best_dict = self.__dict__

        for epoch in range(num_epochs):
            # train loop
            start_time = time.time()

            running_train_loss = 0
            for count, (image, score) in enumerate(train_ds):
                loss = self.train_step(image, score)
                running_train_loss += loss.numpy()
            running_train_loss /= count

            # eval loop
            running_eval_loss = 0
            for count, (image, score) in enumerate(eval_ds):
                loss = self.eval_step(image, score)
                running_eval_loss += loss.numpy()
            running_eval_loss /= count

            epoch_time = time.time() - start_time

            message_str = "Epoch: {}, Train_loss: {:.2f}, Eval_loss: {:.2f}, Took {:.2f} seconds."
            print(message_str.format(epoch, running_train_loss, running_eval_loss, epoch_time))
            # save the losses
            train_loss.append(running_train_loss)
            eval_loss.append(running_eval_loss)

            if running_eval_loss < best_eval_loss:
                best_eval_loss = running_eval_loss
                best_dict = deepcopy(self.__dict__)  # brutal
            else:
                tolerance += 1

            if tolerance > early_stop_tolerance:
                self.__dict__ = best_dict
                break

        print('Training finished')
        return train_loss, eval_loss

    @tf.function
    def train_step(self, images, scores):
        """
        :param images: (n, h, w, c)
        :param scores: (n,)
        :return loss
        """
        with tf.GradientTape() as tape:
            predictions = self.forward(images)
            loss = self.loss_fun(scores, predictions, self.lmd, self.weights)

        grads = tape.gradient(loss, self.weights)
        self.optim.apply_gradients(zip(grads, self.weights))

        return loss

    @tf.function
    def eval_step(self, images, scores):
        """
        :param images: (n, h, w, c)
        :return: loss
        """
        predictions = self.forward(images)
        loss = self.loss_fun(scores, predictions)

        return loss

    def forward(self, x):
        n, h, w, c = x.shape
        x = tf.cast(x, dtype=tf.float32)

        num_conv_layers = len(self.filter_shapes)
        num_dense_layers = len(self.dense_shapes)

        for i in range(num_conv_layers):
            x = tf.nn.conv2d(input=x,
                             filters=self.weights[i],
                             strides=self.cnn_strides[i],
                             padding='VALID')
            x = tf.nn.leaky_relu(features=x, alpha=self.alpha)
            x = tf.nn.max_pool2d(input=x,
                                 ksize=self.pool_shapes[i],
                                 strides=self.pool_strides[i],
                                 padding='VALID')

            if self.dropout_rate > 0:
                x = tf.nn.dropout(x, self.dropout_rate)

            if self.batch_reg:
                mean, var = tf.nn.moments(x, [0, 1, 2])
                x = tf.nn.batch_normalization(x, mean, var,
                                              self.weights[num_conv_layers+num_dense_layers+i],
                                              self.weights[2*num_conv_layers+num_dense_layers+i],
                                              self.epsilon)

        x = tf.reshape(x, shape=(n, -1))

        for i in range(len(self.dense_shapes)):
            x = tf.matmul(x, self.weights[num_conv_layers+i]),

        x = tf.nn.leaky_relu(x, alpha=self.alpha)

        return x

    def init_weights(self, in_shapes, in_name):
        weights = []
        for i, w_shape in enumerate(in_shapes):
            weight = tf.Variable(initial_value=self.initializer(w_shape),
                                 name='{}_{}'.format(in_name, i),
                                 trainable=True,
                                 dtype=tf.float32)
            weights.append(weight)
        return weights


