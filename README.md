# ML project: Road Segmentation 
Project 2 of the Fall 2020 Machine Learning In the context of the road segmentation challenge proposed in the scope of the Machine Learning course of the
EPFL on aicorwd.com, our team proposed a deep-learning solution based on a U-Net architecture, considered to be the current state-of-the-art for segmentation problems. By implementing pre and post-processing steps, and using the Adam optimiser with a Focal Loss, we were able to obtain a satisfactory model capable of obtaining a F1-score of 0.886 and accuracy of 0.940 on our test set.
## Team Members
- Marijn VAN DER MEER
- Léo MEYNENT
- Vincent TOURNIER

## Pre-trained model

Our model is unfortunately too voluminous to fit inside this repository. You can either train it yourself (which will not guarantee you will obtain the exact same results), or download it from [here](https://drive.google.com/file/d/1GRVuMKb1ED2hkeH5vnD1ygVMw3JCMnjZ/view?usp=sharing).

## Structure of the repository: 
The dataset is available on the official aicrowd.com page [here](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).
- **source**: helper functions for our main script; detailed documentation is available inside the folder
- **data**:
  - **test**: test set data 
  - **train**: training and validation set data 
- `run.py`: script to load `ourmodel.h5` or train a model and do predictions

## Instructions to run:

Python modules requirements:
- numpy (works with 1.18.5) 
- PIL (works with 7.0.0)
- matplotlib (works with 3.2.2)
- scipy (works with 1.4.1)
- sklearn (works with 0.22.2.post1)
- tensorflow (works with V2.4.0)

Predictions will be saved in `data/predictions/` and in `submit.csv`. To reproduce our best score with logistic regression that we submitted on [AIcrowd](https://www.aicrowd.com):
```
python run.py
```

The following optional arguments are available:
- ```-l``` to load from `ourmodel.h5` - the model needs to be downloaded and place in the root of the folder beforehand! (default behaviour)
- ```-t``` to train the model from scratch
- ```-s``` to specify the name of the output submission file
