import tensorflow as tf


class ConvBatchNorm:
    def __init__(self, depth, epsilon):
        self.mean = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
        self.variance = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
        self.beta = tf.Variable(tf.constant(0.0, shape=[depth]))
        self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))

        self.epsilon = epsilon
        self.ema_trainer = tf.train.ExponentialMovingAverage(decay=0.5)

    def update_trainer(self):
        self.ema_trainer.apply([self.mean, self.variance])

    def normalize(self, x, mode='train'):
        if mode == 'train':
            mean, variance = tf.nn.moments(x, [0, 1, 2])
            self.mean.assign(mean)
            self.variance.assign(variance)
            x = tf.nn.batch_normalization(x, self.mean, self.variance,
                                          self.beta, self.gamma, self.epsilon)
        elif mode == 'test':
            mean = self.ema_trainer.average(self.mean)
            variance = self.ema_trainer.average(self.variance)
            local_beta = tf.identity(self.beta)
            local_gamma = tf.identity(self.gamma)
            x = tf.nn.batch_normalization(x, mean, variance,
                                          local_beta, local_gamma, self.epsilon)
        return x

