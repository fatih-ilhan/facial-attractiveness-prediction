import time
from copy import deepcopy

import tensorflow as tf

from models.batch_norm import ConvBatchNorm


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
        self.filter_shapes = [[5, 5, 3, 8],
                              [5, 5, 8, 16],
                              [3, 3, 16, 16],
                              [3, 3, 16, 16]]
        self.dense_shapes = [[784, 64],
                             [64, 1]]
        self.pool_shapes = [1, 5, 1, 3]
        self.cnn_strides = [3, 1, 1, 1]
        self.pool_strides = [1, 2, 1, 2]
        self.weight_shapes = self.filter_shapes + self.dense_shapes

        # arrange batch-norm params
        # initialize weights according to initializer
        self.weights = self.init_weights(self.weight_shapes, 'weight')
        self.batch_norm_list = []
        if self.batch_reg:
            for filter_shapes in self.filter_shapes:
                batch_norm = ConvBatchNorm(depth=filter_shapes[-1], epsilon=0.00001)
                self.batch_norm_list.append(batch_norm)

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
                tolerance = 0
            else:
                tolerance += 1

            if tolerance > early_stop_tolerance or epoch == num_epochs - 1:
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
            predictions = self.forward(images, mode='train')
            loss = loss_fun(scores, predictions, self.lmd, self.weights)

        grads = tape.gradient(loss, self.weights +
                              [b.beta for b in self.batch_norm_list] +
                              [b.gamma for b in self.batch_norm_list])
        self.optim.apply_gradients(zip(grads,
                                       self.weights +
                                       [b.beta for b in self.batch_norm_list] +
                                       [b.gamma for b in self.batch_norm_list]))
        return loss

    @tf.function
    def eval_step(self, images, scores, loss_fun):
        """
        :param images: (n, h, w, c)
        :param scores: (n,)
        :param loss_fun: loss function
        :return: loss
        """
        predictions = self.forward(images, mode='test')
        loss = loss_fun(scores, predictions)

        return loss

    def forward(self, x, mode='train'):
        x = tf.cast(x, dtype=tf.float32)

        num_conv_layers = len(self.filter_shapes)
        num_dense_layers = len(self.dense_shapes)

        for i in range(num_conv_layers):
            x = tf.nn.conv2d(input=x,
                             filters=self.weights[i],
                             strides=self.cnn_strides[i],
                             padding='SAME')
            x = tf.nn.leaky_relu(features=x, alpha=self.alpha)
            x = tf.nn.max_pool2d(input=x,
                                 ksize=self.pool_shapes[i],
                                 strides=self.pool_strides[i],
                                 padding='SAME')

            if self.batch_reg:
                batch_norm = self.batch_norm_list[i]
                x = batch_norm.normalize(x, mode=mode)
                batch_norm.update_trainer()

        x = tf.reshape(x, shape=(tf.shape(x)[0], -1))

        for j in range(num_dense_layers):
            if mode == 'train':
                x = tf.nn.dropout(x, self.dropout_rate)

            x = tf.matmul(x, self.weights[num_conv_layers+j])

            if j == num_dense_layers - 1:
                x = tf.nn.relu(x)
            else:
                x = tf.nn.leaky_relu(features=x, alpha=self.alpha)

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
        labels = tf.cast(tf.squeeze(labels), tf.int32)
        rounded_preds = tf.cast(tf.squeeze(preds), tf.int32)
        rounded_preds = tf.clip_by_value(rounded_preds, 1, 8)
        loss = tf.reduce_mean(tf.abs(labels - rounded_preds))
        return loss
