# ML_Project_2: Project Road Segmentation
Project 2 of the Fall 2020 Machine Learning course. For this choice of project task, they provided a set of satellite images acquired 
from GoogleMaps. They also provided ground-truth images where each pixel is labeled as road or background. Our task was to train a classifier to segment roads in these images, i.e. assigns a label `road=1, background=0` to each pixel.

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
