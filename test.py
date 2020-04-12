import os
import tensorflow as tf
import matplotlib.pyplot as plt

from train import loss_l1


def test(model, test_ds, **params):
    model.loss_fun = loss_l1
    running_eval_loss = 0
    for count, (image, score) in enumerate(test_ds):
        loss = model.eval_step(image, score)
        running_eval_loss += loss.numpy()
    running_eval_loss /= count

    print("Test l1 loss: {}".format(running_eval_loss))


def test_sample(model, data_dir):

    sample_img_path = os.path.join(data_dir, 'sample_dir')
    sample_imgs = os.listdir(sample_img_path)

    for img_name in sample_imgs:
        img_path = os.path.join(sample_img_path, img_name)

        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # add batch dimension
        image = tf.expand_dims(image, 0)

        # predict the score
        pred = model.forward(image)
        pred = pred.numpy()

        show(image, pred)


def show(image, score):
    plt.figure()
    plt.imshow(image)
    plt.title(score.numpy().decode('utf-8'))
    plt.axis('off')


