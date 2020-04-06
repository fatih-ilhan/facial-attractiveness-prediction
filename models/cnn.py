import sys
import tensorflow as tf

this = sys.modules[__name__]

layers = [
    ['conv2d', {'weights': [5, 5, 3, 16], 'stride_size': 3}],
    ['leaky_relu', {'alpha': 0.2}],
    ['maxpool', {'pool_size': 5, 'stride_size': 1}],
    ['conv2d', {'weights': [5, 5, 16, 16], 'stride_size': 1}],
    ['leaky_relu', {'alpha': 0.2}],
    ['maxpool', {'pool_size': 5, 'stride_size': 1}],
    ['conv2d', {'weights': [3, 3, 16, 32], 'stride_size': 1}],
    ['leaky_relu', {'alpha': 0.2}],
    ['maxpool', {'pool_size': 3, 'stride_size': 1}],
    ['conv2d', {'weights': [3, 3, 32, 32], 'stride_size': 1}],
    ['leaky_relu', {'alpha': 0.2}],
    ['maxpool', {'pool_size': 3, 'stride_size': 1}],
    ['flatten', {}],
    ['dense', {'weights': [1152, 256]}],
    ['dense', {'weights': [256, 1]}],
    ['leaky_relu', {'alpha': 0.2}],
]


def relu(inputs):
    return tf.nn.relu(inputs)


def leaky_relu(inputs, alpha):
    return tf.nn.leaky_relu(inputs, alpha=alpha)


def conv2d(inputs, weights, stride_size):
    return tf.nn.conv2d(inputs, weights, strides=[1, stride_size, stride_size, 1], padding='VALID')


def maxpool(inputs, pool_size, stride_size):
    return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding='VALID',
                            strides=[1, stride_size, stride_size, 1])


def dense(inputs, weights):
    return tf.matmul(inputs, weights)


def dropout(inputs, rate):
    return tf.nn.dropout(inputs, rate=rate)


def flatten(inputs):
    return tf.reshape(inputs, shape=(tf.shape(inputs)[0], -1))


def get_weight(initializer, shape, name):
    return tf.Variable(initializer(shape), name=name, trainable=True, dtype=tf.float32)


def get_weights(initializer):
    weights = []
    for i, layer in enumerate(layers):
        if 'weights' in layer[1].keys():
            weights.append(get_weight(initializer, layer[1]['weights'], 'weight{}'.format(i)))
    return weights


def model(x, weights):
    x = tf.cast(x, dtype=tf.float32)
    j = 0
    for layer in layers:
        layer_type = layer[0]
        layer_kwargs = layer[1]
        method_to_call = getattr(this, layer_type)

        if 'weights' in layer_kwargs.keys():
            weight = weights[j]
            j += 1
            layer_kwargs['weights'] = weight

        x = method_to_call(x, **layer_kwargs)

    return x


def loss(pred, target):
    return tf.losses.mean_squared_error(target, pred)
