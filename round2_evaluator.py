import os
import json
import pandas as pd
import numpy as np

from round2.training import train_and_eval
from math import log10, floor
import argparse

class Paths:
    SUBMISSION_DATA_PATH = os.getenv('SUBMISSION_DATA_PATH', './example_data/example_embeddings.npy')
    LABELS_PATH = os.getenv('LABELS_PATH', './example_data/example_labels.npy')
    SPLIT_INFO_FILE = os.getenv('SPLIT_INFO_FILE', './example_data/example_split.json')
    TASK_INFO_FILE = os.getenv('TASK_INFO_FILE', 'round2/metadata/jax_tasks.json')
    CLIP_LENGTHS_FILE = os.getenv('TASK_INFO_FILE', 'round2/metadata/clip_lengths_mouse_triplets.json')
    FRAME_NUMBER_MAP = os.getenv('FRAME_MAP_FILE', 'round2/metadata/mouse_frame_number_map.npy')
    LOG_PATH = os.getenv('TRAINING_LOG_PATH', './temp')
    SHORT_RUN = False

def round_sig(x, sig=4):
    if not x == 0:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    else:
        return x

def validate_submission(submission_file_path, embedding_max_size, frame_map_file):

    submission = np.load(submission_file_path)

    frame_map = np.load(frame_map_file, allow_pickle=True).item()

    if not isinstance(submission, np.ndarray):
        print("Embeddings should be a numpy array")
        return False
    elif not len(submission.shape) == 2:
        print("Embeddings should be 2D array")
        return False
    elif not submission.shape[1] <= embedding_max_size:
        print("Embeddings too large, max allowed is 128")
        return False
    elif not isinstance(submission[0, 0], np.float32):
        print(f"Embeddings are not float32")
        return False

    
    total_clip_length = frame_map[list(frame_map.keys())[-1]][1]
            
    if not len(submission) == total_clip_length:
        print(f"Emebddings length doesn't match submission clips total length")
        return False

    if not np.isfinite(submission).all():
        print(f"Emebddings contains NaN or infinity")
        return False
    
    print("All checks passed")
    del submission

class AIcrowdEvaluator:
    def __init__(self, ground_truth_path, task_name='flies', **kwargs):
        Paths.LABELS_PATH = ground_truth_path
        self.task_name = task_name
        print('Starting mabe task', task_name)

    def merge_ant_scores(self, results_df):
        tasks_to_merge = ['reapplied', 'gasterless', 'locc', 'locc_applied', 'histerid', 'nitidulid', 'platy', 'thethered']
        merge_df = results_df[results_df['Task ID'].apply(lambda tid: tid in tasks_to_merge)]
        results_df = results_df[results_df['Task ID'].apply(lambda tid: tid not in tasks_to_merge)]
        priv_score = merge_df['Private Score'].mean()
        pub_score = merge_df['Public Score'].mean()
        new_df =  pd.DataFrame({'Task ID': ["interactor_type"], "Private Score": [priv_score], "Public Score": [pub_score], "Metric": ["f1_score"]})
        results_df = pd.concat([new_df, results_df], ignore_index=True)
        return results_df

    def get_results(self):
        results_df = pd.read_csv(os.path.join(Paths.LOG_PATH, 'results.csv'))

        if self.task_name == 'antbeetle':
            results_df = self.merge_ant_scores(results_df)
        
        return results_df

    def _evaluate(self, client_payload, _context={}): # pylint: disable=W0102
        submission_file_path = client_payload["submission_file_path"]
        Paths.SUBMISSION_DATA_PATH  = submission_file_path

        if not os.path.exists(Paths.LOG_PATH):
            os.mkdir(Paths.LOG_PATH)
        if self.task_name == 'mouse':
            Paths.TASK_INFO_FILE = 'round2/metadata/mouse_round2_tasks.json'
            Paths.SPLIT_INFO_FOLDER = 'round2/metadata/mouse_round2_splits/'
            Paths.FRAME_NUMBER_MAP = os.getenv('FRAME_MAP_FILE', 
                                          'round2/metadata/mouse_round2_frame_number_map.npy')
            embedding_max_size = 128
            test_size = 0.1
            
        elif self.task_name == 'antbeetle':
            Paths.TASK_INFO_FILE = 'round2/metadata/antbeetle_tasks.json'
            Paths.SPLIT_INFO_FOLDER = 'round2/metadata/antbeetle_splits/'
            Paths.FRAME_NUMBER_MAP = os.getenv('FRAME_MAP_FILE', 
                                          'round2/metadata/antbeetle_frame_number_map.npy')
            embedding_max_size = 128
            test_size = 0.1

        elif self.task_name == 'mouse_behaviors':
            Paths.TASK_INFO_FILE = './round2/metadata/mouse_behaviors_tasks.json'
            Paths.SPLIT_INFO_FOLDER = './round2/metadata/mouse_new_splits/split_files/'
            Paths.FRAME_NUMBER_MAP = './round2/metadata/mouse_new_splits/frame_number_map.npy'
            embedding_max_size = 128
            test_size = 0.1
        
        # validate_submission(Paths.SUBMISSION_DATA_PATH, embedding_max_size, Paths.FRAME_NUMBER_MAP)

        print("Starting training")
        train_and_eval.run_all_tasks(Paths, test_size, client_payload["subsample_factor"])
    
        results = self.get_results()

        return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='mouse', choices=['mouse', 'antbeetle', 'mouse_behaviors'])
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--submission', type=str, default=None)
    parser.add_argument('--labels', type=str, default=None)
    parser.add_argument('--subsample_factor', type=int, default=1)

    args = parser.parse_args()

    if args.task == 'mouse':
        task = 'mouse'
        dataf = 'mouse_triplets'
    elif args.task == 'antbeetle':
        task = 'antbeetle'
        dataf = 'ant_beetles'
    elif args.task == 'mouse_behaviors':
        task = 'mouse_behaviors'
        dataf = 'mouse_triplets'

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    ## Local testing
    # ground truth labels
    if args.labels is None:
        labels_file = f'/home/dipam/aicrowd/mabe2022/data/round2_upload/{dataf}/submission_labels.npy'
    else:
        labels_file = args.labels
    evaluator = AIcrowdEvaluator(labels_file, task_name=task)
    # submission file
    if args.submission is None:
        sub_file = f'/home/dipam/aicrowd/mabe2022/data/round2_upload/{dataf}/sample_submission.npy'
    else:
        sub_file = args.submission
    client_payload = {"submission_file_path": sub_file, "subsample_factor": args.subsample_factor}
    results = evaluator._evaluate(client_payload)

    print(results)

    results.to_csv(os.path.join(args.output_dir, task + '-round-2-' + '-' 
                                                + sub_file.split('/')[-1] + '.csv'), 
                   index=False)


