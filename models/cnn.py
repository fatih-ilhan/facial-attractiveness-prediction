import tensorflow as tf


class CNN:
    def __init__(self, **params):
        # parse params
        self.initializer = params['initializer']
        self.optim = params['optimizer']
        self.loss_fun = params['loss_function']
        self.alpha = params['alpha']

        # set filter, pool and dense shapes
        self.weight_shapes = [[5, 5, 3, 16],
                              [5, 5, 16, 16],
                              [3, 3, 16, 32],
                              [3, 3, 32, 32],
                              [1152, 256],
                              [256, 1]]
        self.pool_shapes = [5, 5, 3, 3]

        # initialize weights according to initializer
        self.weights = self.init_weights(self.weight_shapes, 'weight')

    def fit(self, train_ds, eval_ds, num_epochs):
        """
        :param train_ds: :obj:tf.Dataset
        :param eval_ds: :obj:tf.Dataset
        :param num_epochs: int
        :return: train_loss, eval_loss
        """
        print('Training starts...')
        train_loss = []
        eval_loss = []
        for epoch in range(num_epochs):
            # train loop
            running_train_loss = 0
            for count, (image, score) in enumerate(train_ds):
                loss = self.train_step(image, score)
                running_train_loss += loss
            running_train_loss /= count

            # eval loop
            running_eval_loss = 0
            for count, (image, score) in enumerate(eval_ds):
                loss = self.eval_step(image, score)
                running_eval_loss += loss
            running_eval_loss /= count

            message_str = "Epoch: {}, Train_loss: {:.2f}, Eval_loss: {:.2f}"
            print(message_str.format(epoch, running_train_loss, running_eval_loss))

            # save the losses
            train_loss.append(running_train_loss)
            eval_loss.append(running_eval_loss)

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
            loss = self.loss_fun(scores, predictions)
        grads = tape.gradient(loss, self.weights)
        self.optim.apply_gradients(zip(grads, self.weights))

        return loss.numpy()

    @tf.function
    def eval_step(self, images, scores):
        """
        :param images: (n, h, w, c)
        :return: loss
        """
        predictions = self.forward(images)
        loss = self.loss_fun(scores, predictions)

        return loss.numpy()

    def forward(self, x):
        n, h, w, c = x.shape
        x = tf.cast(x, dtype=tf.float32)

        for i in range(4):
            x = tf.nn.conv2d(input=x, filters=self.weights[i], strides=3)
            x = tf.nn.leaky_relu(features=x, alpha=self.alpha)
            x = tf.nn.max_pool2d(input=x, ksize=self.pool_shapes[i], strides=1)

        x = tf.reshape(x, shape=(n, -1))
        x = tf.matmul(x, self.weights[4])
        x = tf.matmul(x, self.weights[5])
        x = tf.leaky_relu(x, alpha=self.alpha)

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


