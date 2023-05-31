# Evaluator code for MABe 2022 Challenge

This repo is based on [mabe-2022-public-evaluator](..%2Fmabe-2022-public-evaluator%2FREADME.md)

This repository contains evaluator code for Round 1 (trajectory data) and Round 2 (video data) of the Multi Agent Behavior Challenge (MABe) 2022.

**Challenge page** - 
https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022

## Data usage (Trajectory data)

1. Download train.npy and test.npy. Train is used for training and validation, and test only contains trajectory data (no labels) for prediction.
2. See the getting started notebooks for an example of how to load in the train & test files, and generating `submission.npy`.

Mouse Triplet: https://www.aicrowd.com/showcase/getting-started-mabe-2022-mouse-triplets-round-1

Fly Groups: https://www.aicrowd.com/showcase/getting-started-mabe-challenge-2022-fruit-flies-v-0-2-2kb

## Data evaluation (Trajectory data)

1. Download the ground truth labels (test_labels.npy) from https://data.caltech.edu/records/20186
2. Extract and place them under the data folder.
3. Change the path links for the ground truth files under `round1_evaluator.py` (at the end of the file under `__main__`)
4. Place your `submission.npy` file in the data folder.
5. Change the path link for the submission file.
6. Set the required task name under for the round you are trying to evaluate.
7. Run `python round1_evaluator.py` for Round 1 or `python round2_evaluator.py` for Round 2

## Python version and packages

Originally used with Python 3.8.13 - But should work with any python version above 3.6

Originally used packages

```
numpy==1.21.0
scikit-learn==1.0.2
pandas==1.3.5
tqdm==4.60.0
```

## Evaluator details

The internal flow of the submissions is described [here](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022#submission).

**Training details** - All models trained use linear models using Scikit-Learn using ridge regression. `Ridge` for regression tasks and `RidgeClassifier` for binary classification tasks. Additionally three seeds are trained for every model where the seed is used to split the dataset 90/10 for training and validation. For classification tasks, the `class_weights` parameter is set to `balanced` for both rounds.

**Scoring** - For binary tasks, predictions are taken via 2/3 vote. For regression tasks, predictions are averaged over all seeds. Once predictions are merged, the score calculated with MSE for regression tasks and F1 score for classification tasks.

## Contributors

- Jennifer J. Sun (Caltech)
- Ann Kennedy (Northwestern University)
- Kristin Branson (Howard Hughes Medical Institute)
- Markus Marks (Caltech)
- Dipam Chakraborty (AIcrowd)
