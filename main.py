import pickle as pkl

from config import Config
from dataset import ImageDataset
from train import train


def main():
    model_name = 'cnn'
    data_folder = 'data'

    config_obj = Config(model_name)

    print("Starting experiments")
    for exp_count, conf in enumerate(config_obj.conf_list):
        print('\nExperiment {}'.format(exp_count))
        print('-*-' * 10)
        datasets = ImageDataset(data_dir=data_folder,
                                batch_size=conf["batch_size"],
                                transform_flags=conf['transform_flags'])
        train(datasets.train_ds, datasets.val_ds, exp_count, **conf)


if __name__ == '__main__':
    main()
