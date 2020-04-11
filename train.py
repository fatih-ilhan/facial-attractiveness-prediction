import os
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt
from models.cnn import CNN


def loss_l2(y_true, y_pred, weights=None):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def loss_l1(y_true, y_pred, weights=None):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def train(train_ds, val_ds, exp_count, **params):
    results_dir = 'results_'
    save_dir = os.path.join(results_dir, str(exp_count))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    num_epochs = params["num_epochs"]
    learning_rate = params["learning_rate"]
    optimizer_type = params["optimizer_type"]
    initializer_type = params["initializer_type"]
    loss_type = params["loss_type"]
    alpha = params["alpha"]

    optimizer_dispatcher = {"adam": tf.optimizers.Adam}
    initializer_dispatcher = {"glorot": tf.initializers.glorot_uniform}
    loss_dispatcher = {"l2": loss_l2, "l1": loss_l1}

    optimizer = optimizer_dispatcher[optimizer_type](learning_rate)
    initializer = initializer_dispatcher[initializer_type]()
    loss_fun = loss_dispatcher[loss_type]

    model = CNN(initializer=initializer,
                optimizer=optimizer,
                loss_function=loss_fun,
                alpha=alpha)

    train_loss, eval_loss = model.fit(train_ds, val_ds, num_epochs)

    # plot and save the loss curve
    plot_loss_curve(train_loss, eval_loss, loss_type, save_dir)

    save_dict = params
    save_dict['train_loss'] = train_loss
    save_dict['eval_loss'] = eval_loss
    print('Saving...')
    save_path = os.path.join(save_dir, 'config.pkl')
    pkl.dump(save_dict, open(save_path, 'rb'))


def plot_loss_curve(train_loss, eval_loss, loss_type, save_dir):
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, '-o')
    plt.plot(range(len(eval_loss)), eval_loss, '-o')
    plt.title('Learning Curve')
    plt.xlabel('num_epoch')
    plt.ylabel('{} loss'.format(loss_type))
    plt.grid(True)
    plt.savefig(save_path, dpi=400)
