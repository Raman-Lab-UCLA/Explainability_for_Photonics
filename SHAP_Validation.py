#Import SHAP Values
with open('C:/Users/cyyeu/Documents/Python/CNN/shap_explanations.data', 'rb') as filehandle:
    shap_values = pickle.load(filehandle)

# Convert Original Background Image to New Image
back_img = loadImages(back_img_path)
back_img = np.array(back_img)[0]
shap_values_convert = shap_values[47].squeeze()
for i in range(np.shape(shap_values_convert)[0]):
    for j in range(np.shape(shap_values_convert)[1]):
        if np.max(shap_values_convert) > shap_values_convert[i][j] > np.max(shap_values_convert)*0.05: #95% red        
            if back_img[i][j] == 255: 
                new_image[i][j] = 0
            elif 255 > back_img[i][j] >= 0:
                new_image[i][j] = 255        
    else:
            new_image[i][j] = back_img[i][j]

# Plot and Save New Image
plt.imshow(new_image,cmap='gray')
plt.imsave('SHAP_Validation.png',new_image,cmap='gray')