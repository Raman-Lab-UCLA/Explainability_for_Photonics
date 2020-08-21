# Explainability_for_Metasurfaces
<p align="center">
  <img src="https://github.com/Raman-Lab-UCLA/Explainability_for_Metasurfaces/blob/master/artwork/explainability_publication.PNG" width="800" />
</p>

## Introduction
Welcome to the Raman Lab GitHub! This repo will walk you through the code used in the following publication: https://pubs.acs.org/doi/full/10.1021/acsphotonics.0c01067 

Here, we use Deep SHAP (or SHAP) to explain the behavior of nanophotonic structures learned by a convolutional neural network (CNN). 

## Requirements
Python3 and Tensorflow 1.0 are recommended due to incompatibility issues with SHAP (as of this writing). 

Installation and usage directions for Deep SHAP are at: https://github.com/slundberg/shap

## Steps
### 1) Train the CNN (CNN_Train.py)
Download the files in the 'Training Data' folder and update the following lines in the 'CNN_Train.py' file:
```python
## Define Data (Images and Spectra) File Locations
img_path = 'C:/.../*.png'
spectra_path = 'C:/.../Spectra.csv'
save_dir = 'C:/.../model.h5'
```
Running this file will train the CNN and save the model in the specified location. 

### 2) Explain CNN Results
Deep SHAP explains the predictions of an input image in reference to a 'background'. This background can be a collection of images or a single image. To minimize noise, our recommendation is to use a single image, or a 'white' image. This will compare the importance of a feature, to the absence of this feature, towards a target output. 

### 3) Explanation Validation
To validate that the explanations represent physical phenomena, we used the SHAP explanations to reconstruct the original image, which can either suppress or enhance an absorption spectrum. This reconstructed image can be imported directly into EM simulation software (e.g., Lumerical FDTD)
