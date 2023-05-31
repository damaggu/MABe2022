import os
import sklearn
from sklearn.linear_model import RidgeClassifier, Ridge
import numpy as np
import json
import pickle

from round2.training.dataloader import MABeDataSplitter

class MABeSingleTaskTrainer:
    def __init__(self, 
                 data_splitter):
        self.data_splitter = data_splitter

    def split_data(self, seed, split_keys, test_size, task_id):
        self.data_splitter.load_split_info(task_id)
        self.data_splitter.split_and_load_data(seed, split_keys, test_size)

    def setup_logging(self, log_path, train_prefix):
        assert os.path.exists(log_path)
        self.training_id = f'{train_prefix}_{self.data_splitter.task_id}_seed{self.data_splitter.seed}'
        self.log_path = os.path.join(log_path, self.training_id)
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.model_path = os.path.join(self.log_path, f'model_{self.training_id}.sav')
    
    def setup_neural_net(self, alpha=1.0):
        model_cls, _ = self._get_model_params_for_task(alpha=alpha)

        model = model_cls()

        self.model = model

    def _get_model_params_for_task(self, alpha=1.0):
        if self.data_splitter.task_type == 'Discrete':
            model_cls = lambda *args: RidgeClassifier(*args, class_weight='balanced', alpha=1.0)
            metric_fn = self.f1_score
        else:
            model_cls = lambda *args: Ridge(*args, alpha=alpha)
            metric_fn = self.mse_score

        return model_cls, metric_fn

    def get_agg_and_metric(self):
        _, metric_fn = self._get_model_params_for_task()
        if self.data_splitter.task_type == 'Discrete':
            metric_name = 'f1_score'
            agg_fn = self._binary_most_repeated
        else:
            metric_name = 'mean_mse'
            agg_fn = self._mean_list_of_arrays

        return agg_fn, metric_fn, metric_name

    def _mean_list_of_arrays(self, array_list):
        return np.mean(np.concatenate([array_list]), axis=0)
    
    def _binary_most_repeated(self, array_list):
        return np.float32(np.mean(np.concatenate([array_list]), axis=0) > 0)

    def train(self):
        X = self.data_splitter.X_train
        y = self.data_splitter.y_train

        self.model.fit(X, y)

        pickle.dump(self.model, open(self.model_path, 'wb'))
    
    def f1_score(self, y_true, y_pred):
        return sklearn.metrics.f1_score(y_true, y_pred,average='binary')

    def mse_score(self, y_true, y_pred):
        return sklearn.metrics.mean_squared_error(y_true, y_pred)
    
    def load_model(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def evaluate(self, val=True):
        if val:
            X = self.data_splitter.X_val
            y_true = self.data_splitter.y_val
        else:
            X = self.data_splitter.X_train
            y_true = self.data_splitter.y_train

        _, metric_fn = self._get_model_params_for_task()
        y_pred = self.model.predict(X)
        return metric_fn(y_true, y_pred)
        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import json
    with open("./example_data/example_split.json", 'r') as fp:
        split_info = json.load(fp)

    mds = MABeDataSplitter(submission_data_path='./example_data/example_embeddings.npy',
                           labels_path='./example_data/example_labels.npy',
                           task_id='time_of_day',
                           split_info=split_info,
                           split_keys=['publicTest', 'privateTest'],
                           seed=42,
                           test_size=0.1)

    trainer = MABeSingleTaskTrainer(data_splitter=mds)

    trainer.setup_logging(log_path='./temp/', train_prefix='')
    trainer.setup_neural_net()
    trainer.train()
    print("Evalution Score", trainer.evaluate())
