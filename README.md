# SGAE implementation in PyTorch

## Project details
This repository contains a PyTorch implementation of the score-guided autoencdoer (SG-AE) presented in "Enhancing Unsupervised Anomaly Detection with Score-guided Network"

## Requirements
- numpy==1.20
- pandas==1.1
- python==3.8.5
- pytorch==1.5.1
- sklearn==0.24
- tqdm==4.6

## Documentation
- data: file path for datasets
- dataloader.py: definition of dataloader class, including data preprocess. 
- sgae_main.py: entrance of experiments, including parameter adjustment.
- sgae_train.py: definition of training and testing. 
- sgae.py: definition of model.

## How to deploy
```
python main.py --[Parameter set] 
```

## Links to datasets
-  tabular data
    - [attack](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
    - [bcsc](https://www.bcsc-research.org/data/rfdataset)
    - [Creditcard](https://www.kaggle.com/mlg-ulb/creditcardfraud) 
    - [diabetic](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
    - [Donor](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose)
    - [intrusion](https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data)
    - [market](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- document
    - [reuters/20news](https://github.com/dmzou/RSRAE)
- image
    - [mnist](https://github.com/zc8340311/RobustAutoencoder)

