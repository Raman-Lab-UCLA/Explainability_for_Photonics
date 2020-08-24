# Explainability for Photonics
<p align="center">
  <img src="https://github.com/Raman-Lab-UCLA/Explainability_for_Photonics/blob/master/artwork/explainability_publication.PNG" width="800" />
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
Deep SHAP explains the predictions of an 'Base' image in reference to a 'Background'. This Background can be a collection of images or a single image. To minimize noise, our recommendation is to use a 'white' image as the Base, and the image to be evaluated as the Background. This will compare the importance of a feature, to the absence of this feature, towards a target output. Simply update the following paths and run the 'SHAP_Explanation.py' script (you can refer to the <b>Examples</b> folder for sample Background and Base images):
```python
## Define File Locations (CNN, Test Image, and Background Image)
model = load_model('C:/.../model.h5', compile=False)
back_img_path = 'C:/.../Background.png'
base_img_path = 'C:/.../Base.png'
```
After running the script, a list of SHAP value heatmaps (<b>shap_values</b>) will be generated. The size and order of this list reflects the CNN's outputs, and the resolution of the heatmaps are the same as the CNN input images. Therefore, to plot a specific heatmap (corresponding to a particular wavelength), simply index the list as such:
```python
shap.image_plot(shap_values[i], back_img.reshape(1,40,40,1), show=False) #where 'i' a value between 0 and the total list size
```
<p align="center">
  <img src="https://github.com/Raman-Lab-UCLA/Explainability_for_Photonics/blob/master/artwork/shap_values_index.png" width="250" />
</p>

Optionally, for ease of viewing, the SHAP values can be normalized and replotted like so: 
```python
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors

with open('C:/.../shap_explanations.data', 'rb') as filehandle:
    shap_values = pickle.load(filehandle)
    
X = np.arange(-20, 20, 1)
Y = np.arange(-20, 20, 1)
X, Y = np.meshgrid(X, Y)

maximum = np.max(shap_values)
minimum = -np.min(shap_values)

shap_i = shap_values[i][:][:][:][:] #where 'i' a value between 0 and the total list size
shap_i[shap_i>0] = shap_i[shap_i>0] / maximum
shap_i[shap_i<0] = shap_i[shap_i<0] / minimum
shap_values_normalized = shap_i.squeeze()[::-1]

fig = plt.figure()
ax = fig.gca()
pcm = ax.pcolormesh(X, Y, shap_values_normalized, norm=colors.SymLogNorm(linthresh=0.01, linscale=1),cmap='bwr', vmin=-1, vmax = 1)
fig.colorbar(pcm)
ax.axis('off')
```
<p align="center">
  <img src="https://github.com/Raman-Lab-UCLA/Explainability_for_Photonics/blob/master/artwork/shap_values_replot.png" width="200" />
</p>

### 3) Explanation Validation (SHAP_Validation.py)
To validate that the explanations represent physical phenomena, we used the SHAP explanations to reconstruct the original image, which can either suppress or enhance an absorption spectrum. This reconstructed image can be imported directly into EM simulation software (e.g., Lumerical FDTD). Run the 'SHAP_Validation.py' script after specifying the location of the saved SHAP values:
```python
#Import SHAP Values
with open('C:/.../shap_explanations.data', 'rb') as filehandle:
    shap_values = pickle.load(filehandle)
```
Tune the conversion settings in the script by modifying the following line in the script:
```python
if np.max(shap_values_convert) > shap_values_convert[i][j] > np.max(shap_values_convert)*0.05: #Convert Top 95% of Red Pixels        
```
<p align="center">
  <img src="https://github.com/Raman-Lab-UCLA/Explainability_for_Photonics/blob/master/artwork/shap_validation.PNG" width="500" />
</p>
