import os
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt
from models.cnn import CNN


def train(train_ds, val_ds, exp_count, overwrite_flag, **params):

    if overwrite_flag:
        tag = exp_count
    else:
        tag_list = [int(path.split("_")[-1]) for path in os.listdir("results") if "exp" in path]
        tag = max(tag_list) + 1

    save_dir = 'experiment_' + str(tag)
    save_dir = os.path.join('results', save_dir)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    num_epochs = params["num_epochs"]
    learning_rate = params["learning_rate"]
    optimizer_type = params["optimizer_type"]
    initializer_type = params["initializer_type"]
    loss_type = params["loss_type"]
    early_stop_tolerance = params["early_stop_tolerance"]

    optimizer_dispatcher = {"adam": tf.optimizers.Adam,
                            "sgd": tf.optimizers.SGD,
                            "rmsprop": tf.optimizers.RMSprop}
    initializer_dispatcher = {"xavier": tf.initializers.glorot_uniform,
                              "random": tf.random_normal_initializer}
    loss_dispatcher = {"l2": loss_l2, "l1": loss_l1}

    optimizer = optimizer_dispatcher[optimizer_type](learning_rate)
    initializer = initializer_dispatcher[initializer_type]()
    loss_fun = loss_dispatcher[loss_type]

    model = CNN(initializer=initializer,
                optimizer=optimizer,
                loss_function=loss_fun,
                **params)

    train_loss, val_loss, evaluation_val_loss = model.fit(train_ds, val_ds, num_epochs, early_stop_tolerance)

    # plot and save the loss curve
    plot_loss_curve(train_loss, val_loss, loss_type, save_dir)

    save_dict = params
    save_dict['train_loss'] = train_loss
    save_dict['val_loss'] = val_loss
    save_dict['evaluation_val_loss'] = evaluation_val_loss
    print('Saving...')
    conf_save_path = os.path.join(save_dir, 'config.pkl')
    model_save_path = os.path.join(save_dir, 'model.pkl')
    for path, obj in zip([conf_save_path, model_save_path], [save_dict, model]):
        with open(path, 'wb') as file:
            pkl.dump(obj, file)


def loss_l2(y_true, y_pred, lmd=0, weights=None):
    loss = tf.reduce_mean(tf.square(tf.squeeze(y_true) - tf.squeeze(y_pred)))
    if weights is not None and lmd > 0:
        # Calculate regularization term by finding norm of weights
        reg_term = 0
        for w in weights:
            reg_term += tf.nn.l2_loss(w)
        loss += reg_term * lmd
    return loss


def loss_l1(y_true, y_pred, lmd=0, weights=None):
    loss = tf.reduce_mean(tf.abs(tf.squeeze(y_true) - tf.squeeze(y_pred)))
    if weights is not None and lmd > 0:
        # Calculate regularization term by finding norm of weights
        reg_term = 0
        for w in weights:
            reg_term += tf.nn.l2_loss(w)
        loss += reg_term * lmd
    return loss


def plot_loss_curve(train_loss, eval_loss, loss_type, save_dir):
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, '-o')
    plt.plot(range(len(eval_loss)), eval_loss, '-o')
    plt.title('Learning Curve')
    plt.xlabel('num_epoch')
    plt.ylabel('{} loss'.format(loss_type))
    plt.legend(['train', 'validation'])
    plt.grid(True)
    plt.savefig(save_path, dpi=400)
