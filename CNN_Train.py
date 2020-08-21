import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob
import cv2
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras.optimizers import Adam

## Define Data (Images and Spectra) File Locations
img_path = 'C:/Users/cyyeu/Documents/MATLAB/Apps/maxwellfdfd-master/_test_Chris/Images/Lumerical/Images/Images/*.png'
spectra_path = 'C:/Users/cyyeu/Documents/MATLAB/Apps/maxwellfdfd-master/_test_Chris/Data/absorptionData_More_80ptsV2.csv'
save_dir = 'C:/Users/cyyeu/Documents/Python/CNN/model_10k_batchsize16_epochs300_FULLDATASET_MORE80ptsV2.h5'

## load Images (CNN Input)
def loadImages(path):
    loadedImages = []
    filesname = glob.glob(path)
    filesname.sort()
    for imgdata in filesname:
        if os.path.isfile(os.path.splitext(os.path.join(path, imgdata))[0] + ".png"):
            img_array = cv2.imread(os.path.join(path, imgdata))
            img_array = np.float32(img_array)
            img_size = 40
            new_array = cv2.resize(img_array, (img_size, img_size))
            gray = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)       
            loadedImages.append(gray)
    return loadedImages

imgs = loadImages(img_path)
CNN_input = np.array(imgs).reshape(len(imgs),40,40,1)

## Load Spectra from Excel (CNN Output)
CNN_output = np.array(np.float32(pd.read_csv(spectra_path, header = 0, index_col=0)))

# Split Data into Train and Test Sets
CNN_input_train, CNN_input_test, CNN_output_train, CNN_output_test = train_test_split(CNN_input, CNN_output, test_size = 0.1, random_state = 42)
print('data size after spliting')
print('CNN_input_train size: {}'.format(np.shape(CNN_input_train)))
print('CNN_input_test size: {}'.format(np.shape(CNN_input_test)))
print('CNN_output_train size: {}'.format(np.shape(CNN_output_train)))
print('CNN_output_test size: {}'.format(np.shape(CNN_output_test)))

# Define CNN Architecture
model = Sequential()
model.add(Conv2D(16, (3,3), padding = 'same', input_shape = (40,40,1)))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2))
model.add(AveragePooling2D(pool_size = (2,2), strides = 2))
model.add(Conv2D(32, (3,3), padding = 'same'))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2))
model.add(AveragePooling2D(pool_size = (2,2), strides = 2))
model.add(Conv2D(64, (3,3), padding = 'same'))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2))
model.add(AveragePooling2D(pool_size = (2,2), strides = 2))
model.add(Conv2D(128, (3,3), padding = 'same'))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2))
model.add(AveragePooling2D(pool_size = (2,2), strides = 2))
model.add(Flatten())
model.add(Dense(80))

cnnopt = Adam()
model.compile(loss = 'mean_squared_error', optimizer = cnnopt, metrics = ['accuracy'])
print(model.summary())        

# Train and Save CNN
epochs = 300
batch_size = 16
validation_data = (CNN_input_test, CNN_output_test)
model.fit(CNN_input_train, CNN_output_train, batch_size = batch_size, epochs = epochs, validation_data = validation_data)
score = model.evaluate(CNN_input_test, CNN_output_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(model.summary())
model.save(save_dir)

