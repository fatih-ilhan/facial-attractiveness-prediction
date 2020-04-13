import os
import tensorflow as tf
import matplotlib.pyplot as plt


def test(model, test_ds):
    test_loss = model.step_loop(test_ds, model.eval_step, model.loss_fun_evaluation)
    print("Test Rounded MAE loss: {:.3f}".format(test_loss))


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


