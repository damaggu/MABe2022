# Challenge Design

## Finalizing the Problem Statement

This include deciding and finalizing the exact task that the participants needs to perform. 

## Finalizing the Evaluation Metrics

This include deciding the evaluation metrics that the participant submission will be judged on. 

We recommend deciding atleast 2 metrics with one being the primary one and the other secondary. 

We support leaderboard sorting based on upto 2 metrics, primary and secondary, but we can still show any number of metrics on leaderboard or the participant, if required.

## Finalizing the Dataset

This include finalizing the dataset and freezing it for the challenge purpose. 

The dataset should also be split into:
- train set: which will be given to the participant along with the labels of it.
- validation set: This will also be given to the participant with its labels. This is optional though.
- test/evaluation set: The evaluation will happen on this dataset and therefore the leaderboard will be based on the performance of participant on this dataset. Data points without the target varible may or may not be given to the participant based on the challenge requirements. 

# Dataset

## Splitting the test set into Public/Private Score

We provide the feature of releaving the metric scores of partial test data during the challenge and full test data only after the challenge ends. We highly recommend this as it helps in avoiding overfitting and leaderboard probing. 

## Upload the dataset to Cloud

Once the dataset is finalised, it should be uploaded on AIcrowd cloud service so that it can easily be accessed by the participants and can also be uploaded in Resources section of challenge page. 

## Upload it in Resources section on Challenge page

Once uploaded on cloud it should be added to the resources section of the challenge page, so that it becomes easily available for participants.

# Evaluator

## Setting up the Evaluator

Configuring the evaluation framework properly is the most important part for challenge execution. There are many small details such as timeouts, resources etc that need to be explored and adjusted. 
These things that should be finalised before setting up the evalutor:

- Timeouts for
    - Training, if we are providing
    - Inference
- Specifying the machine used for evaluation purpose
- Media for Leaderboard
- Public/Private Scores
- Should the public score be used in the final ranking?

These questions are also asked in config.md which needs to be answer before settng up of evaluator. 

## Setting up the Media for Leaderboard

We can show any kind of media on our leaderboard and if possible recommend showing some media that can be generated using participants predictions code. 

Few examples, 
- [Video](https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge/leaderboards)
- [GIF](https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge/leaderboards)

## Testing the Evaluator

Once the evaluator is set up, it should be thoroughly tested for normal as well edges cases to avoid any problem in future. 

## Expected Compute Cost

On the basis of evaluator testing, the total cost that is expected to incur for the evaluation of submission assuming the expected number of submissions should be calculated and shared with the team. 

# Starter Kit

The starter kit is the first piece of code that the participants interact with respect to the challenge. It should be simple and detailed enough to provide all the information needed for the participant to get started. 

## Creation of Starter Kit

The starter kit with random prediction model should be created either in the form of notebook in case of CSV or Notebook submission type challenge or in the form of a gitlab repo in case of code based submission challenge. 

Example:
- [CSV](https://www.aicrowd.com/showcase/getting-started-code-for-task-1-classical-classification)
- [Notebook Submission](https://www.aicrowd.com/showcase/getting-started-notebook-for-nlp-feature-engineering)
- [Gitlab Code Submission](https://github.com/AIcrowd/food-recognition-benchmark-starter-kit)

## Testing of Starter Kit by making a submission.

The created starter kit should be tested by making a submission to the challenge. 

## Adding it to the challenge page

Once the starter kit is finalised it should be linked at appropiate place and also added to the notebook section if its a notebook.

# Baseline

Baselines in addition to the starter kit provide great value from a participant's point of view. They help participants to build on top on something which is much easier and hence increases the activity in the challenge.

## Creation of Baseline

Same as the starter kit, baseline should be created in the same format. 

Example:
- [CSV](https://www.aicrowd.com/showcase/baseline-notebook-for-task-1-classical-classification)
- [Notebook Submission](https://www.aicrowd.com/showcase/getting-started-notebook-for-nlp-feature-engineering)
- [Gitlab Code Submission](https://github.com/AIcrowd/food-recognition-benchmark-starter-kit)

## Testing of Baseline by making a submission.

The created baseline should be tested by making a submission to the challenge. 

## Adding it to the challenge page

Once the baseline is finalised it should be linked at appropiate place and also added to the notebook section if its a notebook.

# Challenge Communication

Challenge communications play a major role in a success of a challenge. It is through these announcements and mails that the users know about the challenge in the first place. 

## Creation of Marketing/Communication Calendar

Creation of a detailed email campaign calendar for the whole duration the challenge goes through starting from the launch of the challenge to the winners' announcements. 

Some of the emails that should be there are
- Launch Email
- Email to nudge participants after 1 week of challenge launch
- Mid-Challenge Nudge Mail
- Last X Days Left
- Challenge Completion & Winners Announcement Mail

## Launch Mail

Creation of the email that will go out to all the users of AIcrowd or a targetted segment for the launch of the challenge. 

## Social Media Post

Creation of the launch social media posts that will posted from various social media handles of AIcrowd for the launch of the challenge. 

We post on the following platforms
- Facebook
- Twitter
- LinkedIn

## Discord Launch Announcement

Creation of the announcemnet that will go out on AIcrowd Discord for the launch of the challenge. 

## Discourse Launch Announcement

Creation of the announcemnet that will go out on AIcrowd Discourse for the launch of the challenge. 

# Challenge Testing

Keeping the persona of a beginner user, someone should go through the challenge page and make a submission the starter kit and the instructions present in it. 

## Going through the Overview

Review the challenge page and make sure things are correct and make sense. In case of any complex world, it should be linked to some external link explaining it. 

## Testing of Starter Kit by making a submission

The created starter kit should be tested by making a submission to the challenge. 

## Testing of Baseline by making a submission

The created baseline should be tested by making a submission to the challenge. 


## Checking the score on leaderboard

After making the submission, the score on the leaderboard should be checked for correctness it as well as the correctness and right information of leaderboard metrics and its sorting order. 


