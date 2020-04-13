import time
from copy import deepcopy

import tensorflow as tf


class CNN:
    def __init__(self, **params):
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
        self.dense_shapes = [[1152, 1]]
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

    def fit(self, train_ds, val_ds, num_epochs, early_stop_tolerance):
        """
        :param train_ds: :obj:tf.Dataset
        :param val_ds: :obj:tf.Dataset
        :param num_epochs: int
        :return: train_loss, eval_loss
        """
        print('Training starts...')
        train_loss = []
        val_loss = []

        tolerance = 0
        best_epoch = 0
        best_val_loss = 1e6
        evaluation_val_loss = best_val_loss
        best_dict = self.__dict__

        for epoch in range(num_epochs):
            # train and validation loop
            start_time = time.time()
            running_train_loss = self.step_loop(train_ds, self.train_step, self.loss_fun)
            running_val_loss = self.step_loop(val_ds, self.eval_step, self.loss_fun)
            epoch_time = time.time() - start_time

            message_str = "Epoch: {}, Train_loss: {:.3f}, Validation_loss: {:.3f}, Took {:.3f} seconds."
            print(message_str.format(epoch + 1, running_train_loss, running_val_loss, epoch_time))
            # save the losses
            train_loss.append(running_train_loss)
            val_loss.append(running_val_loss)

            if running_val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = running_val_loss
                best_dict = deepcopy(self.__dict__)  # brutal
            else:
                tolerance += 1

            if tolerance > early_stop_tolerance:
                self.__dict__ = best_dict
                evaluation_val_loss = self.step_loop(val_ds, self.eval_step, self.loss_fun_evaluation)
                message_str = "Early exiting from epoch: {}, Rounded MAE for validation set: {:.3f}."
                print(message_str.format(best_epoch, evaluation_val_loss))
                break

        print('Training finished')
        return train_loss, val_loss, evaluation_val_loss

    @staticmethod
    def step_loop(dataset, step_fun, loss_fun):
        count = 0
        running_loss = 0.0

        for count, (image, score) in enumerate(dataset):
            loss = step_fun(image, score, loss_fun)
            running_loss += loss.numpy()

        running_loss /= (count + 1)

        return running_loss

    @tf.function
    def train_step(self, images, scores, loss_fun):
        """
        :param images: (n, h, w, c)
        :param scores: (n,)
        :param loss_fun: loss function
        :return loss
        """
        with tf.GradientTape() as tape:
            predictions = self.forward(images)
            loss = loss_fun(scores, predictions, self.lmd, self.weights)

        grads = tape.gradient(loss, self.weights)
        self.optim.apply_gradients(zip(grads, self.weights))

        return loss

    @tf.function
    def eval_step(self, images, scores, loss_fun):
        """
        :param images: (n, h, w, c)
        :param scores: (n,)
        :param loss_fun: loss function
        :return: loss
        """
        predictions = self.forward(images)
        loss = loss_fun(scores, predictions)

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

            x = tf.nn.dropout(x, self.dropout_rate)

            if self.batch_reg:
                mean, var = tf.nn.moments(x, [0, 1, 2])
                x = tf.nn.batch_normalization(x, mean, var,
                                              self.weights[num_conv_layers+num_dense_layers+i],
                                              self.weights[2*num_conv_layers+num_dense_layers+i],
                                              self.epsilon)

        x = tf.reshape(x, shape=(tf.shape(x)[0], -1))

        for i in range(len(self.dense_shapes)):
            x = tf.matmul(x, self.weights[num_conv_layers+i])
            x = tf.nn.dropout(x, self.dropout_rate)

        x = tf.nn.leaky_relu(x, alpha=self.alpha)
        tf.print(x)

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

    @staticmethod
    def loss_fun_evaluation(labels, preds):
        """
        :param labels:
        :param preds:
        :return:
        """
        labels = tf.cast(labels, tf.int32)
        rounded_preds = tf.cast(preds, tf.int32)
        rounded_preds = tf.clip_by_value(rounded_preds, 1, 8)
        loss = tf.reduce_mean(tf.abs(labels - rounded_preds))
        return loss


