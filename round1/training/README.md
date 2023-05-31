# Competition flow

# Submission Format

Participants are to submit embeddings for the entire test set provided in one numpy file.

The maxmimum embedding size allowed is 256.

NaN values are not allowed in the embedding submitted by the participant.

# Training pipeline for tasks

Each task is trained with the provided embeddings independently.

## Base neural net trained per task

- Keras 2 layer neural network
- Fixed number of epochs per task, save model with best validation loss
- Activation and loss Function
    - Sigmoid Binary Crossentropy for binary labels
    - tanh with output transform for continuous labels, mse loss


## Hyperparameter search

A grid search of 4 values of the following parameters

- Learning rate - [0.1, 0.03, 0.01, 0.001]
- Hidden layer units - [32, 100, 200, 512]

Total 16 runs per task, the model with lowest validation loss is retained. 

## Multiple seeds

3 seeds are used for splitting the data (90-10 train/validation). For each seed, the full hyperparameter search is performed. Scores of 3 models with best validation loss are averaged.

A total of 16*3 = 48 training runs are done per task.

# Data format

There are 3 data folders for data that will be used for training and evaluation on the competition servers

Each folder contains labels for training or evauation of the models

- **submission_train** - This contains the labels of snippets used for training.
- **public_train** - This contains the labels of snippets used for evaluation and reporting of the public leaderboard.
- **private_train** - This contains the labels of snippets used for evaluation of the private leaderboard which will be revealed at the end of the competition and used for selecting the winners.

The [`example_data`](example_data) folder contains dummy data which follows the format that all the training and evaluation data needs to follow.

The [`example_data/submission_train/snippet_metadata.csv`](example_data/submission_train/snippet_metadata.csv) contains the metadata information of the training data. The columns represent the following:

 Column   |      Contents      | 
|----------|:-------------:|
| task_id |  Unique id is to be used for every task |
| snippet_id |  Unique id is to be used for every snippet  |
| path | full path to the label data file |
| start_idx | The starting index of the snippet in the combined submission file with all the embeddings |
| end_idx | The ending index of the snippet in the combined submission file with all the embeddings |
| snippet_size | number of frames in the snippet |
| label_type | One of 'discrete' or 'continious'. 'discrete' denotes binary 0 or 1, 'continious' denotes any value between 0 and 1 |

Each label file should be a 1D numpy array of labels, either discrete or continious. Saved and loadable with `np.save` and `np.load`. 

**NaN values contained in the label arrays will be removed during training from both the label and embedding array in the corresponding index**

