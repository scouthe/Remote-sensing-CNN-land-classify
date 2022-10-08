from glob import glob
import os, math, random, glob, time
random.seed(2)
import numpy as np
import tensorflow as tf
from sklearn.utils import resample
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
#import matplotlib.pyplot as plt


#########################################################################
##### PART - B: READING NUMPY ARRAYS. TRAINING AND SAVING THE MODEL #####
#########################################################################

os.chdir(r"F:\PycharmProjects\img_class\Remote-sensing-CNN-land-classify")
out_model='7class'
# Load arrays from .npy files
features = np.load('CNN_3by3_features.npy')
labels = np.load('CNN_3by3_labels.npy')

# Separate and balance the classes
# forest
forest_features = features[labels==0]
forest_labels = labels[labels==0]
# water
water_features = features[labels==1]
water_labels = labels[labels==1]
# gress
gress_features = features[labels==2]
gress_labels = labels[labels==2]
# ground
ground_features = features[labels==3]
ground_labels = labels[labels==3]

built_features = features[labels==4]
built_labels = labels[labels==4]
# road
road_features = features[labels==5]
road_labels = labels[labels==5]

out_features = features[labels==6]
out_labels = labels[labels==6]

print('Number of records in each class:')
print('forest: %d, water: %d, gress:%d' % (forest_features.shape[0], water_features.shape[0],gress_features.shape[0]))


# Downsample the majority class
forest_features = resample(forest_features,
                             replace = False, # sample without replacement
                             n_samples = gress_features.shape[0], # match minority n
                             random_state = 2)

forest_labels = resample(forest_labels,
                           replace = False, # sample without replacement
                           n_samples = gress_labels.shape[0], # match minority n
                           random_state = 2)


ground_features = resample(ground_features,
                            replace = False, # sample without replacement
                            n_samples = gress_features.shape[0], # match minority n
                            random_state = 2)

ground_labels = resample(ground_labels,
                          replace = False, # sample without replacement
                          n_samples = gress_labels.shape[0], # match minority n
                          random_state = 2)
# print('Number of records in balanced classes:')
# print('Built: %d, Unbuilt: %d' % (built_labels.shape[0], water_labels.shape[0]))

print('resample')
print('forest: %d, water: %d, gress:%d, groud:%d, build:%d, road:%d' % (forest_features.shape[0], water_features.shape[0],gress_features.shape[0],ground_features.shape[0],built_features.shape[0],road_features.shape[0]))
# Combine the balanced features
features = np.concatenate((forest_features, water_features,gress_features,ground_features,built_features,road_features,out_features), axis=0)
labels = np.concatenate((forest_labels, water_labels,gress_labels,ground_labels,built_labels,road_labels,out_labels), axis=0)
forest_sum=forest_features.shape[0]
water_sum=water_features.shape[0]
gress_sum=gress_features.shape[0]
ground_sum=ground_features.shape[0]
build_sum=built_features.shape[0]
road_sum=road_features.shape[0]
out_sum=out_features.shape[0]

feature_pos=[forest_sum,water_sum,gress_sum,ground_sum,build_sum,road_sum,out_sum]

# Normalise the features  图片最大值为2047，应当除以2047进行归一化，以图片最大值，不是特征最大值
print(features.min(),features.max())
features = features*4 / (255.0*7)

print('New values in input features, min: %d & max: %d' % (features.min(), features.max()))

def feature_clip(featurepos,trainProp,order=1,labels=None):
    """
    average clip each feature
    :param featurepos: input different feature position
    :param trainProp: rate of trainging dataset
    :param order: default is training dataset, otherwise is test dataset
    :return:
    """
    clip=np.array([])
    i=0
    if order==1:
        for item in featurepos:
            index=np.arange(i,int((item)*trainProp)+i)
            clip=np.concatenate((clip,index),axis=0)
            i+=item
    else:
        for item in featurepos:

            index=np.arange(int((item)*trainProp)+i,item+i)
            clip=np.concatenate((clip,index),axis=0)
            i += item
    return clip

# Define the function to split features and labels
def train_test_split(features, labels, trainProp=0.7):
    dataSize = features.shape[0]
    sliceIndex = int(dataSize*trainProp)
    randIndex = np.arange(dataSize)
    # random.shuffle(randIndex)
    # a0=randIndex[:sliceIndex]
    # a=[randIndex[:sliceIndex]]
    trainx = features[[randIndex[:sliceIndex]], :, :, :][0]
    train=feature_clip(feature_pos,trainProp,1)
    test=feature_clip(feature_pos,trainProp,0)

    train_x=features[[train.astype(np.int32)], :, :, :][0]

    test_x = features[[test.astype(np.int32)], :, :, :][0]
    train_y = labels[train.astype(np.int32)]

    test_y = labels[test.astype(np.int32)]
    return(train_x, train_y, test_x, test_y)

# Call the function to split the data
train_x, train_y, test_x, test_y = train_test_split(features, labels)
print('train data shape')
print(train_x.shape[0])
print('test data shape')
print(test_x.shape[0])
print(train_x.max())

print('Reshaped features:', train_x.shape, test_x.shape)
_, rowSize, colSize, nBands = train_x.shape


# Create a model

model = keras.Sequential()
model.add(Conv2D(32,strides=1, kernel_size=1, padding='valid', activation='relu', input_shape=(rowSize, colSize, nBands)))
model.add(Dropout(0.25))
model.add(Conv2D(64,strides=1, kernel_size=1, padding='valid', activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64,strides=1, kernel_size=1, padding='valid', activation='relu'))
model.add(Dropout(0.25))

# model.add(MaxPooling2D((2,2),strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer= keras.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=20,verbose=1,steps_per_epoch=2)
# model = tf.keras.models.load_model('F:\PycharmProjects\img_class\Landsat-Classification-Using-Convolution-Neural-Network-master\cnn'
#                                    '\\trained_models\\CNN_7class_3by3.h5')
print(model.summary())
# Predict for test data 
yTestPredicted = model.predict(test_x,steps=1)
# yTestPredicted = yTestPredicted[:,-1]
classnum=np.argmax(yTestPredicted,axis=1)


model.save('CNN_%s_3by3.h5' % (out_model))
# Calculate and display the error metrics
# yTestPredicted = (yTestPredicted>0.5).astype(int)
cMatrix = confusion_matrix(test_y,classnum)
# cMatrix = confusion_matrix(test_y, yTestPredicted)
print("Confusion matrix:\n", cMatrix)

# pScore = precision_score(test_y, yTestPredicted,average=None)
pScore = precision_score(test_y, classnum,average=None)
rScore = recall_score(test_y, classnum,average=None)
fScore = f1_score(test_y, classnum,average=None)


    
