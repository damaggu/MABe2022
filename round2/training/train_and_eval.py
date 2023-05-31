import json
import os
import numpy as np
import time
import pandas as pd

from round2.training.dataloader import MABeDataSplitter
from round2.training.trainer import MABeSingleTaskTrainer

from tqdm.auto import tqdm


def train_multiple_tasks(data_splitter, seed, task_id_list, Constants, test_size, alpha_vals):
    start = time.time()
    
    trainer = MABeSingleTaskTrainer(data_splitter=data_splitter)
    
    print(f'Seed {seed}')
    start = time.time()
    
    model_paths = {}
    for task_id in tqdm(task_id_list):
        for alpha in alpha_vals:
            trainer.split_data(seed=seed, 
                            split_keys=['SubmissionTrain'], 
                            test_size=test_size,
                            task_id=task_id)
            trainer.data_splitter.load_labels(task_id=task_id)

            trainer.setup_logging(log_path=Constants.LOG_PATH, train_prefix=f'alpha-{alpha}-')
            trainer.setup_neural_net()
            trainer.train()
            model_paths[task_id + '-alpha-' + str(alpha)] = trainer.model_path

    print(f'Seed {seed}: All tasks train time', time.time() - start)
    start = time.time()

    return model_paths

def predict_single_task_multiseed(data_splitter, task_id, model_paths, Constants, split_keys):
    start = time.time()

    trainer = MABeSingleTaskTrainer(data_splitter=data_splitter)
    trainer.split_data(seed=0, # Doesn't matter when test size is 0
                       split_keys=split_keys, 
                       test_size=0.0,
                       task_id=task_id)
    
    trainer.data_splitter.load_labels(task_id=task_id)

    all_y_preds = []
    y_true = data_splitter.y_train
    X = data_splitter.X_train
    for mp in model_paths:
        trainer.load_model(mp)
        y_pred = trainer.model.predict(X)
        all_y_preds.append(y_pred)
    
    agg_fn, metrics_fn, metric_name = trainer.get_agg_and_metric()
    
    print(f'Task {task_id}: {trainer.data_splitter.split_keys} Predict time for all seeds', time.time() - start)

    return all_y_preds, agg_fn, metrics_fn, y_true, metric_name

def eval_single_task_multiseed(data_splitter, task_id, model_paths, Constants, task_is_sequence_level):

    def rem_nan_idx(y):
        y_notnan_idx = ~np.isnan(y)
        y_new = y[y_notnan_idx]
        return y_new

    res = predict_single_task_multiseed(data_splitter, task_id, model_paths, Constants, split_keys=['publicTest'])
    public_y_preds, agg_fn, metrics_fn, public_y_true, metric_name = res

    public_agg = agg_fn(public_y_preds)
    public_score = metrics_fn(public_y_true, public_agg)

    # Average Pooling Sequences
    # load each sequence with frame number map
    # regenerate not nan indexes by adding frame indexes back to back
    remnan_start = 0
    y_pooled_true, y_pooled_pred = [], []
    for sk in data_splitter.train_snippets:
        start, end = data_splitter.frame_number_map[sk]
        y_orig = data_splitter.labels['label_array'][data_splitter.task_idx, start:end]
        y_remnan = rem_nan_idx(y_orig)
        if len(y_remnan) > 0:
            avg_pool_gt = np.mean(y_remnan)
            if metric_name == 'f1_score':
                avg_pool_gt = int(avg_pool_gt >= 0.5)
            y_pooled_true.append(avg_pool_gt)
            avg_pool_pred = np.mean(public_agg[remnan_start:remnan_start+len(y_remnan)])
            if metric_name == 'f1_score':
                avg_pool_pred = int(avg_pool_pred >= 0.5)
            y_pooled_pred.append(avg_pool_pred)
            remnan_start += len(y_remnan)

    res = predict_single_task_multiseed(data_splitter, task_id, model_paths, Constants, split_keys=['privateTest'])
    private_y_preds, agg_fn, metrics_fn, private_y_true, metric_name = res

    pri_pub_y_preds = [np.concatenate([pr, pb]) for pr, pb in zip(private_y_preds, public_y_preds)]
    pri_pub_y_agg = agg_fn(pri_pub_y_preds)
    pri_pub_y_true = np.concatenate([private_y_true, public_y_true])
    private_score = metrics_fn(pri_pub_y_true, pri_pub_y_agg)

    remnan_start = 0
    private_agg = agg_fn(private_y_preds)
    for sk in data_splitter.train_snippets:
        start, end = data_splitter.frame_number_map[sk]
        y_orig = data_splitter.labels['label_array'][data_splitter.task_idx, start:end]
        y_remnan = rem_nan_idx(y_orig)
        if len(y_remnan) > 0:
            avg_pool_gt = np.mean(y_remnan)
            if metric_name == 'f1_score':
                avg_pool_gt = int(avg_pool_gt >= 0.5)
            y_pooled_true.append(avg_pool_gt)
            avg_pool_pred = np.mean(private_agg[remnan_start:remnan_start+len(y_remnan)])
            if metric_name == 'f1_score':
                avg_pool_pred = int(avg_pool_pred >= 0.5)
            y_pooled_pred.append(avg_pool_pred)
            remnan_start += len(y_remnan)

    single_seed_scores = [metrics_fn(pri_pub_y_true, pred_single_seed) for pred_single_seed in pri_pub_y_preds]
    no_ensemble_score = np.mean(single_seed_scores)

    if task_is_sequence_level:
        pooled_score = metrics_fn(y_pooled_true, y_pooled_pred)
    else:
        pooled_score = -1

    return private_score, public_score,  metric_name, no_ensemble_score, pooled_score, task_is_sequence_level


def run_all_tasks(Constants, test_size, subsample_factor):
    # with open(Constants.SPLIT_INFOS, 'r') as fp:
        # split_info = json.load(fp)

    with open(Constants.TASK_INFO_FILE, 'r') as fp:
        tasks_info = json.load(fp)

    seeds = tasks_info['seeds']
    task_id_list = tasks_info['task_id_list']
    sequence_level_tasks = tasks_info['sequence_level_tasks']
    alpha_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    if Constants.SHORT_RUN:
        seeds = [42, 43]
        task_id_list = task_id_list[1:3]
        alpha_vals = [0.1, 1.0]

    start = time.time()
    data_splitter = MABeDataSplitter(submission_data_path=Constants.SUBMISSION_DATA_PATH,              
                                     split_info_folder=Constants.SPLIT_INFO_FOLDER,
                                     labels_path=Constants.LABELS_PATH,
                                     frame_number_map_file=Constants.FRAME_NUMBER_MAP,
                                     subsample_factor=subsample_factor, )
    print(f'Submission data load time', time.time() - start)

    model_paths_all = {}
    for seed in seeds:
        model_paths_all[seed] = train_multiple_tasks(data_splitter, seed, task_id_list, Constants, test_size, alpha_vals)
    
    results = []
    model_paths = {}
    for task_id in task_id_list:
        alpha_scores = []
        for alpha in alpha_vals:
            model_paths[alpha] = []
            for seed in seeds:
                model_paths[alpha].append(model_paths_all[seed][task_id + '-alpha-' + str(alpha)])
        
            task_is_sequence_level = task_id in sequence_level_tasks
            res = eval_single_task_multiseed(data_splitter, task_id, model_paths[alpha], Constants, task_is_sequence_level)
            alpha_scores.append(res[0].copy())
            if alpha == 1.0:
                private_score, public_score, metric_name, no_ensemble_score, pooled_score, task_is_sequence_level = res
                task_results = [task_id, private_score, public_score, metric_name, no_ensemble_score, pooled_score, task_is_sequence_level]

        print("Results: Task", task_id, "| Metric", metric_name, "| Public", public_score, "| Private", private_score, '\n')
        task_results.extend(alpha_scores)
        results.append(task_results)
    

    columns=['Task ID', 'Private Score', 'Public Score', 'Metric', 
             'No Ensemble Score', 'Pooled Score', 'Sequence Level Task']
    for alpha in alpha_vals:
        columns.append(f"Score Alpha {alpha}")

    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(os.path.join(Constants.LOG_PATH, 'results.csv'), index=False)


if __name__ == '__main__':
    class Constants:
        SUBMISSION_DATA_PATH = os.getenv('SUBMISSION_DATA_PATH', './example_data/example_embeddings.npy')
        LABELS_PATH = os.getenv('LABELS_PATH', './example_data/example_labels.npy')
        SPLIT_INFO_FILE = os.getenv('SPLIT_INFO_FILE', './example_data/example_split.json')
        TASK_INFO_FILE = os.getenv('TASK_INFO_FILE', '../metadata/jax_tasks.json')
        LOG_PATH = os.getenv('TRAINING_LOG_PATH', './temp')

        # SUBMISSION_DATA_PATH = '/home/dipam/aicrowd/mabe2022/data/round1_upload/mouse_triplets/sample_submission.npy' 
        # LABELS_PATH = '/home/dipam/aicrowd/mabe2022/data/round1_upload/mouse_triplets/submission_labels.npy'
        # SPLIT_INFO_FILE = "../metadata/jax_split.json"
        # TASK_INFO_FILE = "../metadata/jax_tasks.json"

        # SUBMISSION_DATA_PATH = '/home/dipam/aicrowd/mabe2022/data/round1_upload/fruit_flies/sample_submission.npy' 
        # LABELS_PATH = '/home/dipam/aicrowd/mabe2022/data/round1_upload/fruit_flies/submission_labels.npy'
        # SPLIT_INFO_FILE = "../metadata/flies_split.json"
        # TASK_INFO_FILE = "../metadata/fly_tasks.json"

    run_all_tasks(Constants)
    
