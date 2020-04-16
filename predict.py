import os

import tensorflow as tf
import matplotlib.pyplot as plt


def test(model, test_ds):
    test_loss = model.step_loop(test_ds, model.eval_step, model.loss_fun_evaluation)
    print("Test Rounded MAE loss: {:.3f}".format(test_loss))


def test_sample(model, data_dir, device, slot):

    sample_img_path = os.path.join(data_dir, 'sample')
    sample_imgs = os.listdir(sample_img_path)

    for img_name in sample_imgs:
        if ".gitkeep" in img_name:
            continue
        label = img_name[0]

        img_path = os.path.join(sample_img_path, img_name)

        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        with tf.device('/' + device + ':' + str(slot)):
            # add batch dimension
            image = tf.expand_dims(image, 0)
            # predict the score
            pred = model.forward(image).numpy()

        show(image, pred, label)
    plt.show()


def show(image, pred, label):
    plt.figure()
    plt.imshow(tf.squeeze(image))
    plt.title("Label:{}, Pred:{}".format(label, round(pred.item())), fontsize=18)
    plt.axis('off')
