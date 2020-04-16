import itertools
from random import shuffle

model_params_pool = {
    "cnn": {
        "batch_size": [64],
        "transform_flags": [[0, 0, 0, 0]],
        "learning_rate": [3e-4],
        "num_epochs": [300],
        "optimizer_type": ["adam"],
        "initializer_type": ["random"],
        "loss_type": ["l2"],
        "alpha": [0.01],
        "batch_reg": [False],
        "lambda": [1e-3],
        "dropout_rate": [0.1],
        "early_stop_tolerance": [5]
    }
}


class Config:
    """
    This object contains manually given parameters
    """
    def __init__(self, model_name):
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
