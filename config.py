import itertools
from random import shuffle

model_params_pool = {
    "cnn": {
        "batch_size": [8],
        "transform_flags": [[0, 0, 1, 0]],
        "learning_rate": [0.01],
        "num_epochs": [50],
        "optimizer_type": ["adam"],
        "initializer_type": ["glorot"],
        "loss_type": ['l1'],
        "alpha": [0.2]
    }
}


class Config:
    """
    This object contains manually given parameters
    """
    def __init__(self, model_name):
        self.experiment_params = {"evaluation_metric": ["mean_squared_error"],
                                  "scale_mode": "none",
                                  "online_flag": False,
                                  "num_epochs": 100,
                                  "early_stop_tolerance": 3,
                                  "train_val_test_ratio": [0.6, 0.2, 0.2]}

        assert len(self.experiment_params["train_val_test_ratio"]) == 3 and \
                   sum(self.experiment_params["train_val_test_ratio"]) == 1

        self.model_name = model_name
        self.model_params = model_params_pool[self.model_name]

        self.conf_list = self.create_params_list(dict(**self.model_params))
        self.num_confs = len(self.conf_list)

    def next(self):
        for conf in self.conf_list:
            yield conf

    @staticmethod
    def create_params_list(pool):
        params_list = []
        keys = pool.keys()
        lists = [l for l in pool.values()]
        all_lists = list(itertools.product(*lists))
        for i in range(len(all_lists)):
            params_list.append(dict(zip(keys, all_lists[i])))
        shuffle(params_list)
        return params_list