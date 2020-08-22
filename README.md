# Explainability_for_Metasurfaces
<p align="center">
  <img src="https://github.com/Raman-Lab-UCLA/Explainability_for_Metasurfaces/blob/master/artwork/explainability_publication.PNG" width="800" />
</p>

## Introduction
Welcome to the Raman Lab GitHub! This repo will walk you through the code used in the following publication: https://pubs.acs.org/doi/full/10.1021/acsphotonics.0c01067 

Here, we use Deep SHAP (or SHAP) to explain the behavior of nanophotonic structures learned by a convolutional neural network (CNN). 

## Requirements
Python3 and Tensorflow 1.0 are recommended due to incompatibility issues with SHAP (as of this writing). 

Installation and usage instructions for Deep SHAP are at: https://github.com/slundberg/shap

## Steps
### 1) Train the CNN (CNN_Train.py)
Download the files in the 'Training Data' folder and update the following lines in the 'CNN_Train.py' file:
```python
## Define File Locations (Images, Spectra, and CNN Model Save)
img_path = 'C:/.../*.png'
spectra_path = 'C:/.../Spectra.csv'
save_dir = 'C:/.../model.h5'
```
Running this file will train the CNN and save the model in the specified location. 

### 2) Explain CNN Results (SHAP_Explanation.py)
Deep SHAP explains the predictions of an 'base' image in reference to a 'background'. This background can be a collection of images or a single image. To minimize noise, our recommendation is to use a 'white' image as the base, and the image to be evaluated as the 'background'. This will compare the importance of a feature, to the absence of this feature, towards a target output. Simply update the following paths with run the 'SHAP_Explanation.py' script:
```python
## Define File Locations (CNN, Test Image, and Background Image)
model = load_model('C:/.../model.h5', compile=False)
test_img_path = 'C:/.../Test.png'
back_img_path = 'C:/.../Background.png'
```

### 3) Explanation Validation
To validate that the explanations represent physical phenomena, we used the SHAP explanations to reconstruct the original image, which can either suppress or enhance an absorption spectrum. This reconstructed image can be imported directly into EM simulation software (e.g., Lumerical FDTD).
