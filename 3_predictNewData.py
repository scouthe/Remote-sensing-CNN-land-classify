import os, math, random
random.seed(2)
import numpy as np
import tensorflow as tf
from pyrsgis import raster

######################################################################
##### PART - C: LOADING THE SAVED MODEL AND PREDICTING NEW IMAGE #####
######################################################################

# Load the saved model
model = tf.keras.models.load_model('F:\PycharmProjects\img_class\Remote-sensing-CNN-land-classify\CNN_7class_3by3.h5')

# Load a new multispectral image
ds, featuresHyderabad = raster.read("F:\PycharmProjects\img_class\Remote-sensing-CNN-land-classify\World_Multi.TIF")

outFile = 'landsat_7Class_CNN_predicted_3by3.tif'

os.chdir(r"F:\PycharmProjects\img_class\Remote-sensing-CNN-land-classify")
# Load arrays from .npy files
features = np.load('CNN_3by3_features.npy')

# 归一化
features = features*4 / (255.0*7)

print('Shape of the new features', features.shape)

# Predict new data and export the probability raster
newPredicted = model.predict(features)
newPredicted = np.argmax(newPredicted,axis=1)

prediction = np.reshape(newPredicted, (ds.RasterYSize, ds.RasterXSize))
raster.export(prediction, ds, filename=outFile, dtype='float')
