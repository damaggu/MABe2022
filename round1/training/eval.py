import os
import json
import tensorflow as tf
import sklearn

from dataloader import MABeDataSplitter
import utils

class MABeSingleTaskEvaluator:
    def __init__(self,
                 task_id,
                 snippet_metadata_path, 
                 submission_data_path,
                 ):
        self.snippet_metadata_path = snippet_metadata_path
        self.taskinfo = utils.TaskInfo(task_id, snippet_metadata_path)

        self.data_splitter = MABeDataSplitter(self.taskinfo, 
                                              submission_data_path,
                                              seed=0,
                                              test_size=0.0)
                                                      
        self.evaluation_function = self._get_evaluation_function_for_task()

    def load_and_evaluate(self, results_path):
        self.load_configs(results_path)
        self.setup_neural_net(self.model_params)
        return self.evaluation_function()

    def load_configs(self, results_path):

        assert os.path.exists(results_path), f"Results folder {results_path} does not exist"

        files = os.listdir(results_path)
        params_file = [f for f in files if 'json' in f][0]
        with open(os.path.join(results_path, params_file), 'r') as fp:
            self.model_params = json.load(fp)

        model_file = [f for f in files if 'h5' in f][0]
        self.model_path = os.path.join(results_path, model_file)

    def load_data(self):
        self.data_splitter.split_and_load_data()
    
    def setup_neural_net(self, model_params):
        input_size = self.data_splitter.X_train.shape[1]
        hidden_size = model_params['hidden_units']

        output_activation, num_outputs = self._get_model_params_for_task()

        model = tf.keras.models.Sequential([
                    tf.keras.layers.Input(input_size),
                    tf.keras.layers.Dense(hidden_size, activation='tanh'),
                    tf.keras.layers.Dense(num_outputs, activation=output_activation)
        ])

        model.load_weights(self.model_path)
        self.model = model

    def _get_model_params_for_task(self):
        if self.taskinfo.label_type == 'discrete':
            activation = 'sigmoid'
            num_outputs = 1
        else:
            raise NotImplementedError

        return activation, num_outputs
    
    def _get_evaluation_function_for_task(self):
        if self.taskinfo.label_type == 'discrete':
            return self.get_f1_score
        else:
            raise NotImplementedError

    def get_f1_score(self):
        inputs = self.data_splitter.X_train
        labels = self.data_splitter.y_train
        predictions = self.model.predict(inputs)
        
        thresh_preds = predictions > 0.5
        f1_score = sklearn.metrics.f1_score(labels, thresh_preds,average='binary')
        return f1_score

if __name__ == '__main__':
    eval = MABeSingleTaskEvaluator(task_id='task1',
                               snippet_metadata_path='./example_data/submission_train/snippet_metadata.csv', 
                               submission_data_path='./example_data/submission_embedding.npy')

    eval.load_data()
    result = eval.load_and_evaluate(results_path='./temp/run11_task1_seed42')
    print("Evaluated score", result)