import os
import random
import matplotlib.pyplot as plt
import numpy as np


def display_random_images(set, count):
    folder_path = os.path.join("data", set)
    image_path_list = os.listdir(folder_path)
    image_path_list = random.sample(image_path_list, count)

    num_rows = int(np.sqrt(count))
    num_cols = int(np.ceil(count / num_rows))
    for i, image_path in enumerate(image_path_list):
        image = plt.imread(os.path.join(folder_path, image_path))
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(f"Randomly selected images from {set} set")
    plt.savefig(os.path.join("figures", f"random_images_{set}"), bbox_inches='tight')


def display_mean_images(set):
    folder_path = os.path.join("data", set)
    image_path_list = os.listdir(folder_path)

    i = 0
    image_array = np.zeros((80, 80, 3))
    for i, image_path in enumerate(image_path_list):
        image = plt.imread(os.path.join(folder_path, image_path)) / 255
        image_array += image

    mean_image = image_array / (i + 1)

    plt.imshow(mean_image)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Mean image for {set} set")
    plt.savefig(os.path.join("figures", f"mean_image_{set}"), bbox_inches='tight')


def display_attractiveness_histograms(set):
    folder_path = os.path.join("data", set)
    image_path_list = os.listdir(folder_path)

    attractiveness_list = []
    for i, image_path in enumerate(image_path_list):
        level = int(image_path[0])
        attractiveness_list.append(level)

    plt.hist(attractiveness_list, bins=range(1, 10), rwidth=0.9)
    plt.xticks(np.arange(1, 9)+0.5, np.arange(1, 9))
    plt.xlabel("Attractiveness level")
    plt.ylabel("Number of samples")
    plt.title(f"Histogram of attractiveness level for {set} set")
    plt.savefig(os.path.join("figures", f"histogram_{set}"), bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    set = "validation"
    count = 4
    mode_list = ["histogram"]

    if "random_images" in mode_list:
        display_random_images(set, count)
    if "mean_images" in mode_list:
        display_mean_images(set)
    if "histogram" in mode_list:
        display_attractiveness_histograms(set)
