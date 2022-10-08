import os, math, random, glob, time
random.seed(2)
import numpy as np
from pyrsgis import raster
from pyrsgis.ml import imageChipsFromFile



#####################################################################
##### PART - A: CREATING AND STORING IMAGE CHIPS AS NUMPY ARRAYS ####
#####################################################################

# Change the working directory
output_directory = r"F:\PycharmProjects\img_class\Remote-sensing-CNN-land-classify"
os.chdir(output_directory)

# define the file names
feature_file = r"World_Multi.tif"
label_file = r"new_class.tif"

# create feature chips using pyrsgis
features = imageChipsFromFile(feature_file, x_size=3, y_size=3)

""" Update: 29 May 2021
Since I added this code chunk later, I wanted to make least 
possible changes in the remaining sections. The below line changes
the index of the channels. This will be undone at a later stage.
"""
# features = np.rollaxis(features, 3, 1)

# read the label file and reshape it
ds, labels = raster.read(label_file)
labels = labels.flatten()

# check for irrelevant values (we are interested in 1s and non-1s)
# labels = (labels == 0).astype(int)


# print basic details
print('Input features shape:', features.shape)
print('\nInput labels shape:', labels.shape)
# print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))
print('Values in input labels, min: %d & max: %d' % (labels.min(), labels.max()))

# Save the arrays as .npy files
np.save('CNN_3by3_features.npy', features)
np.save('CNN_3by3_labels.npy', labels)
print('Arrays saved at location %s' % (os.getcwd()))
