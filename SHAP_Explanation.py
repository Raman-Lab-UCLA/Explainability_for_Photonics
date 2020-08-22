import shap
import cv2
import numpy as np
from keras.models import load_model
import pickle

## Define File Locations (CNN, Test Image, and Background Image)
model = load_model('C:/.../model.h5', compile=False)
back_img_path = 'C:/.../Background.png'
base_img_path = 'C:/.../Base.png'

## load Image Function
def loadImages(path):
    img_array = cv2.imread(path)
    img_array = np.float32(img_array)
    img_size = 40
    new_array = cv2.resize(img_array, (img_size, img_size))
    gray = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
    return gray

## Load Test and Background Image
base_img = loadImages(base_img_path) 
base_img = np.array(base_img)
back_img = loadImages(back_img_path)
back_img = np.array(back_img)

## Perform SHAP Explanations
background = back_img.reshape(1,40,40,1)
e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(base_img.reshape(1,40,40,1))
shap.image_plot(shap_values, base_img.reshape(1,40,40,1), show=False)

## Save SHAP Explanations
with open('shap_explanations.data', 'wb') as filehandle:
    pickle.dump(shap_values, filehandle)
