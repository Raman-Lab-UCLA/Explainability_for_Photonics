# Explainability_for_Metasurfaces

## Introduction
Welcome to the Raman Lab GitHub! This repo will walk you through the code used in the following publication: https://pubs.acs.org/doi/full/10.1021/acsphotonics.0c01067 

Here, we use Deep SHAP (or SHAP) to explain the behavior of nanophotonic structures learned by a convolutional neural network (CNN). 

## Requirements
Python3 and Tensorflow 1.0 are recommended due to incompatibility issues with SHAP (as of this writing). 

Installation and usage directions for Deep SHAP are at: https://github.com/slundberg/shap

## Steps
### 1) Train the CNN
Download the files in the 'Training Data' folder and update the following lines in the 'CNN_Train.py' file:
```python
## Define Data (Images and Spectra) File Locations
img_path = 'C:/.../*.png'
spectra_path = 'C:/.../Spectra.csv'
save_dir = 'C:/.../model.h5'
```
This will train the CNN and save the model in the specified location. 

### 2) Explain CNN Results
