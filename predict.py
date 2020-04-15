import os
import pickle
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt


def test(model, test_ds):
    test_loss = model.step_loop(test_ds, model.eval_step, model.loss_fun_evaluation)
    print("Test Rounded MAE loss: {:.3f}".format(test_loss))


# TODO: kanka bu çalışmadı bende
def test_sample(model, data_dir, device, slot):

    sample_img_path = os.path.join(data_dir, 'sample')
    sample_imgs = os.listdir(sample_img_path)

    for img_name in sample_imgs:
        if ".gitkeep" in img_name:
            continue

        img_path = os.path.join(sample_img_path, img_name)

        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        with tf.device('/' + device + ':' + str(slot)):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default="CPU")  # device CPU/GPU
    parser.add_argument('--slot', type=int, default=0)  # device slot
    parser.add_argument('--model_id', type=int, default=0)  # overwrite previous results

    args = parser.parse_args()

    model_path = os.path.join("results", f"experiment_{args.model_id}", "model.pkl")
    model = pickle.load(open(model_path, "rb"))
    test_sample(model, "data", args.device, args. slot)
