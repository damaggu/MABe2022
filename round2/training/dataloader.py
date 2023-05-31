import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import json
import os

from tqdm import tqdm


class MABeDataSplitter:
    def __init__(self,
                 submission_data_path,
                 labels_path,
                 frame_number_map_file,
                 split_info_folder,
                 subsample_factor=1, ):
        self.submission_data_path = submission_data_path
        self.labels_path = labels_path
        self.frame_number_map_file = frame_number_map_file
        self.split_info_folder = split_info_folder
        self.split_info = None
        self.subsample_factor = subsample_factor

        self.split_clip_names = None

        self.train_snippets, self.test_snippets = None, None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None

        self.load_embeddings_and_labels()

    def load_split_info(self, task_id):
        with open(os.path.join(self.split_info_folder, f'{task_id}.json'), 'r') as fp:
            self.split_info = json.load(fp)

    def load_embeddings_and_labels(self):
        self.frame_number_map = np.load(self.frame_number_map_file, allow_pickle=True).item()
        self.labels = np.load(self.labels_path, allow_pickle=True).item()

        # if embedding is a dir, load all npy files and concat and perform PCA
        try:
            self.sub_emb = np.load(self.submission_data_path)
        except (ValueError, IsADirectoryError):
            if self.submission_data_path.endswith('.npy'):
                self.sub_emb = np.load(self.submission_data_path, allow_pickle=True).item()
            else:
                # is a dir
                # os list dir
                # load all npy files
                # concat
                self.sub_emb = {}
                for f in tqdm(os.listdir(self.submission_data_path)):
                    if f.endswith('.npy'):
                        filename = f.split('/')[-1].split('.')[0]
                        if filename in self.frame_number_map.keys():
                            file = np.load(os.path.join(self.submission_data_path, f), allow_pickle=True)
                            if 'Frame' in self.submission_data_path and 'mouse' in self.submission_data_path:
                                file = file[:-1]
                            self.sub_emb[filename] = file
                            # print(file.shape)
                        else:
                            print(f'file {filename} not in frame_number_map')
            print('Submission data is a dict')

            results = []
            for frame_keys in tqdm(self.frame_number_map.keys()):
                try:
                    file_res = self.sub_emb[frame_keys]
                    # if file_res.shape[0] < 361:
                    if False:
                        print(f'Key {frame_keys} not found in submission data')
                        # remove from labels
                        index_range = self.labels['frame_number_map'].pop(frame_keys)
                        self.labels['label_array'][:, index_range[0]:index_range[1]] = np.nan
                        self.labels['label_array'] = self.labels['label_array'][:,
                                                     ~np.isnan(self.labels['label_array'][0])]
                    else:
                        results.append(file_res)
                except KeyError:
                    print(f'Key {frame_keys} not found in submission data')
                    # remove from labels
                    index_range = self.labels['frame_number_map'].pop(frame_keys)
                    self.labels['label_array'][:, index_range[0]:index_range[1]] = np.nan
                    self.labels['label_array'] = self.labels['label_array'][:, ~np.isnan(self.labels['label_array'][0])]
            results = np.concatenate(results, axis=0)
            self.sub_emb = results
            # reduce embedding from 2048 to 128 dimensions using PCA
            print('Reducing embedding dimensionality')
            pca = PCA(n_components=128)
            self.sub_emb = pca.fit_transform(self.sub_emb)
            # save reduced embeddings
            reduced_embedding_path = self.submission_data_path.replace('.npy', '_reduced.npy')
            np.save(reduced_embedding_path, self.sub_emb)
            print('Submission data is a dict')

        ## to mabe format if necessary
        if len(self.frame_number_map) == 400:
            if not 'mabe_' in self.submission_data_path:
                original_frame_number_map = np.load('round2/metadata/mouse_round2_frame_number_map.npy',
                                                    allow_pickle=True).item()
                new_sub_emb = np.zeros((len(self.frame_number_map) * 1800, self.sub_emb.shape[-1]))
                for k, v in self.frame_number_map.items():
                    start_idx, end_idx = v
                    new_sub_emb[start_idx:end_idx] = self.sub_emb[
                                                     original_frame_number_map[k][0]:original_frame_number_map[k][1]]
                self.sub_emb = new_sub_emb
            else:
                print('Submission data is already in mabe format')

        # subsample if needed
        if not 'mabe_' in self.submission_data_path:
            self.sub_emb = self.sub_emb[::self.subsample_factor]
        self.labels['label_array'] = self.labels['label_array'][:, ::self.subsample_factor]
        new_frame_number_map = {}
        for k, v in self.frame_number_map.items():
            new_frame_number_map[k] = (v[0] // self.subsample_factor, v[1] // self.subsample_factor)
        self.frame_number_map = new_frame_number_map

    def split_and_load_data(self, seed, split_keys, test_size=0.1):
        self.seed = seed
        self.test_size = test_size
        self.split_keys = split_keys

        self.train_snippets, self.test_snippets = None, None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None

        self._select_by_split_keys()
        self._split_snippet_names()
        self.X_train_full, self.X_val_full = self._split_embeddings()

    def load_labels(self, task_id):
        self.task_id = task_id
        self.y_train, self.y_val = self._load_labels()
        self.remove_nan_indexes()

    def _select_by_split_keys(self):
        self.split_clip_names = []
        for ds in self.split_keys:
            self.split_clip_names.extend(self.split_info[ds])

    def _split_snippet_names(self):
        if self.test_size > 0:
            self.train_snippets, self.test_snippets = train_test_split(self.split_clip_names,
                                                                       test_size=self.test_size,
                                                                       random_state=self.seed)
        else:
            self.train_snippets, self.test_snippets = self.split_clip_names, []

    def remove_nan_indexes(self):

        def rem_nan_idx(X, y):
            y_notnan_idx = ~np.isnan(y)
            X_new = X[y_notnan_idx]
            y_new = y[y_notnan_idx]
            return X_new, y_new

        # y_notnan_idx = ~np.isnan(self.y_train)
        # if np.sum(y_notnan_idx) == 0:
        #     import pdb; pdb.set_trace()
        #     print(self.task_idx, self.task_id)
        #     return

        self.X_train, self.y_train = rem_nan_idx(self.X_train_full, self.y_train)
        if self.test_size > 0:
            self.X_val, self.y_val = rem_nan_idx(self.X_val_full, self.y_val)

    def _get_index(self, snippets, length):
        index = np.zeros(length, bool)
        for sk in snippets:
            start, end = self.frame_number_map[sk]
            index[start:end] = True

        return index

    def _split_embeddings(self):
        self.train_index = self._get_index(self.train_snippets, self.sub_emb.shape[0])
        self.val_index = self._get_index(self.test_snippets, self.sub_emb.shape[0])
        X_train = self.sub_emb[self.train_index]
        X_val = self.sub_emb[self.val_index]
        return X_train, X_val

    def _set_task_info(self):
        self.task_idx = self.labels['vocabulary'].index(self.task_id)
        self.task_type = self.labels['task_type'][self.task_idx]

    def _load_labels(self):

        self._set_task_info()

        y_train = self.labels['label_array'][self.task_idx, self.train_index]
        y_val = self.labels['label_array'][self.task_idx, self.val_index]

        return y_train, y_val


if __name__ == '__main__':
    import json

    with open("./example_data/example_split.json", 'r') as fp:
        split_info = json.load(fp)

    mds = MABeDataSplitter(submission_data_path='./example_data/example_embeddings.npy',
                           labels_path='./example_data/example_labels.npy',
                           task_id='chases',
                           split_info=split_info,
                           split_keys=['publicTest', 'privateTest'],
                           seed=42,
                           test_size=0.1)

    mds.split_and_load_data()

    print(mds.X_train.shape, mds.y_train.shape)
