import os
import json
import pandas as pd
import numpy as np

from round1.training import train_and_eval
from math import log10, floor
import argparse

class Paths:
    SUBMISSION_DATA_PATH = os.getenv('SUBMISSION_DATA_PATH', './example_data/example_embeddings.npy')
    LABELS_PATH = os.getenv('LABELS_PATH', './example_data/example_labels.npy')
    SPLIT_INFO_FILE = os.getenv('SPLIT_INFO_FILE', './example_data/example_split.json')
    TASK_INFO_FILE = os.getenv('TASK_INFO_FILE', 'round1/metadata/jax_tasks.json')
    CLIP_LENGTHS_FILE = os.getenv('CLIP_LENGTHS_FILE', 'round1/metadata/clip_lengths_mouse_triplets.json')
    FRAME_NUMBER_MAP = os.getenv('FRAME_MAP_FILE', 'round1/metadata/mouse_frame_number_map.npy')
    LOG_PATH = os.getenv('TRAINING_LOG_PATH', './temp')
    SHORT_RUN = False # For fast testing

def round_sig(x, sig=4):
    if not x == 0:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    else:
        return x

def validate_submission(submission_file_path, clip_lengths_file, embedding_max_size, frame_map_file):

    submission = np.load(submission_file_path, allow_pickle=True).item()
    with open(clip_lengths_file, 'r') as fp:
       clip_lengths = json.load(fp)

    if not isinstance(submission, dict):
      raise ValueError("Submission should be dict")

    frame_map = np.load(frame_map_file, allow_pickle=True).item()

    if 'frame_number_map' not in submission:
        raise ValueError("Frame number map missing")

    for k,v in frame_map.items():
        sv = submission['frame_number_map'][k]
        if not v == sv:
            raise ValueError("Frame number map should be exactly same as provided in frame_number_map.npy in resources")

    if 'embeddings' not in submission:
        raise ValueError('Embeddings array missing')
    elif not isinstance(submission['embeddings'], np.ndarray):
        raise ValueError("Embeddings should be a numpy array")
    elif not len(submission['embeddings'].shape) == 2:
        raise ValueError("Embeddings should be 2D array")
    elif not submission['embeddings'].shape[1] <= embedding_max_size:
        raise ValueError(f"Embeddings too large, max allowed is {embedding_max_size}")
    elif not isinstance(submission['embeddings'][0, 0], np.float32):
        raise ValueError(f"Embeddings are not float32")
    
    total_clip_length = 0
    for key, clip_length in clip_lengths.items():
        start, end = submission['frame_number_map'][key]
        total_clip_length += clip_length
        if not end-start == clip_length:
            raise ValueError(f"Frame number map for clip {key} doesn't match clip length")


    if not len(submission['embeddings']) == total_clip_length:
        raise ValueError(f"Emebddings length doesn't match submission clips total length")
    
    if not np.isfinite(submission['embeddings']).all():
        raise ValueError("Emebddings contains NaN or infinity")
    
    print("All checks passed")
    del submission

class AIcrowdEvaluator:
    def __init__(self, ground_truth_path, task_name='flies', **kwargs):
        Paths.LABELS_PATH = ground_truth_path
        self.task_name = task_name
        print('Starting mabe task', task_name)

    def average_singlefly_tasks(self, results_df):

        contains_singletask = results_df['Task ID'].apply(lambda name: 'singletask' in name)

        allfly_df = results_df[~contains_singletask]
        singlefly_df = results_df[contains_singletask]
        sf_task_keys = np.unique(singlefly_df['Task ID'].apply(lambda name: name.split('_singletask')[0]))

        singlefly_avg = []
        other_column_keys = list(singlefly_df.columns)[4:]
        for tk in sf_task_keys:
            task_df = singlefly_df[singlefly_df['Task ID'].apply(lambda name: tk in name)]
            public_score = task_df['Public Score'].mean()
            private_score = task_df['Private Score'].mean()
            others_vals = []
            for ck in other_column_keys:
                if 'Score' not in ck:
                    others_vals.append(task_df[ck].values[0])
                else:
                    others_vals.append(task_df[ck].mean())
            singlefly_avg.append((tk, private_score, public_score, 'f1_score', *others_vals))

        singlefly_avg_df = pd.DataFrame(singlefly_avg, columns=singlefly_df.columns)
        results_df_avg = allfly_df.append(singlefly_avg_df)
        results_df_avg = results_df_avg.reset_index(drop=True)

        return results_df_avg

    def get_results(self):
        results_df = pd.read_csv(os.path.join(Paths.LOG_PATH, 'results.csv'))
        if self.task_name == 'flies':
            results_df = self.average_singlefly_tasks(results_df)

        return results_df

    def _evaluate(self, client_payload, _context={}):
        submission_file_path = client_payload["submission_file_path"]
        Paths.SUBMISSION_DATA_PATH  = submission_file_path

        if not os.path.exists(Paths.LOG_PATH):
            os.mkdir(Paths.LOG_PATH)
        if self.task_name == 'mouse':
            Paths.TASK_INFO_FILE = 'round1/metadata/jax_tasks.json'
            Paths.SPLIT_INFO_FILE = 'round1/metadata/jax_split.json'
            Paths.CLIP_LENGTHS_FILE = os.getenv('CLIP_LENGTHS_FILE', 
                                          'round1/metadata/clip_lengths_mouse_triplets.json')
            Paths.FRAME_NUMBER_MAP = os.getenv('FRAME_MAP_FILE', 
                                          'round1/metadata/mouse_frame_number_map.npy')
            embedding_max_size = 128
            test_size=0.1
            
        elif self.task_name == 'flies':
            Paths.TASK_INFO_FILE = 'round1/metadata/fly_tasks.json'
            Paths.SPLIT_INFO_FILE = 'round1/metadata/flies_split.json'
            Paths.CLIP_LENGTHS_FILE = os.getenv('CLIP_LENGTHS_FILE', 
                                          'round1/metadata/clip_lengths_fruit_flies.json')
            Paths.FRAME_NUMBER_MAP = os.getenv('FRAME_MAP_FILE', 
                                          'round1/metadata/flies_frame_number_map.npy')
            embedding_max_size = 256
            test_size = 0.1
        
        validate_submission(Paths.SUBMISSION_DATA_PATH, Paths.CLIP_LENGTHS_FILE, embedding_max_size, Paths.FRAME_NUMBER_MAP)

        print("Starting training")
        train_and_eval.run_all_tasks(Paths, test_size)
    
        results = self.get_results()

        return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='mouse', choices=['mouse', 'flies'])
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--submission', type=str, default=None)
    parser.add_argument('--labels', type=str, default=None)

    args = parser.parse_args()

    if args.task == 'mouse':
        task = 'mouse'
        dataf = 'mouse_triplets'
    elif args.task == 'flies':
        task = 'flies'
        dataf = 'fruit_flies'

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    ## Local testing
    # ground truth labels
    if args.labels is None:
        labels_file = f'/home/dipam/aicrowd/mabe2022/data/round1_upload/{dataf}/submission_labels.npy'
    else:
        labels_file = args.labels

    evaluator = AIcrowdEvaluator(labels_file, task_name=task)
    # submission file
    if args.submission is None:
        sub_file = f'/home/dipam/aicrowd/mabe2022/data/round1_upload/{dataf}/sample_submission.npy'
    else:
        sub_file = args.submission
    client_payload = {"submission_file_path": sub_file}
    results = evaluator._evaluate(client_payload)

    print(results)

    results.to_csv(os.path.join(args.output_dir, task + '-round-1-' + '-' 
                                                + sub_file.split('/')[-1] + '.csv'), 
                   index=False)


