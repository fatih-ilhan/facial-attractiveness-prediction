import os
import pickle as pkl

from config import Config
from dataset import ImageDataset
from train import train
from test import test


def select_best_model(results_dir):
    experiments = os.listdir(results_dir)

    best_model = None
    best_conf = {}
    best_loss = 99
    for exp in experiments:
        exp_path = os.path.join(results_dir, exp)
        conf_path = os.path.join(exp_path, 'config.pkl')
        model_path = os.path.join(results_dir, 'model.pkl')

        with open(conf_path, 'rb') as f:
            config = pkl.load(f)
        eval_loss = config['eval_loss'][-1]
        if eval_loss < best_loss:
            best_loss = eval_loss
            with open(model_path, 'rb') as f:
                best_model = pkl.load(f)
            best_conf = config

    return best_model, best_conf


def main():
    model_name = 'cnn'
    data_folder = 'data'
    results_folder = 'results'

    config_obj = Config(model_name)

    print("Starting experiments")
    for exp_count, conf in enumerate(config_obj.conf_list):
        print('\nExperiment {}'.format(exp_count))
        print('-*-' * 10)
        datasets = ImageDataset(data_dir=data_folder,
                                batch_size=conf["batch_size"],
                                transform_flags=conf['transform_flags'])
        train(datasets.train_ds, datasets.val_ds, exp_count, **conf)

    best_model, best_conf = select_best_model(results_folder)
    datasets = ImageDataset(data_dir=data_folder,
                            batch_size=best_conf["batch_size"],
                            transform_flags=best_conf['transform_flags'])

    print("Testing with best model")
    test(best_model, datasets.test_ds)


if __name__ == '__main__':
    main()
