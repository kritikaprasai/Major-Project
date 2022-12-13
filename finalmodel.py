#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul 25 12:53:08 2019
@author: anu
"""

#import necessary modules
from imutils import paths
import random
import shutil
import os

#Model+dataset Config
print(os.listdir("../data/fold1"))
BASE_PATH = "../data/fold1"
# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2


# take the paths to all input images in the original input directory and shuffle them()
imagePaths = list(paths.list_images(BASE_PATH))
random.seed(42)
random.shuffle(imagePaths)

#computing the training and testing split
i = int(len(imagePaths) * TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

i = int(len(trainPaths) * VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# definig datasets that we will be using to build our modle
datasets = [
	("training", trainPaths, TRAIN_PATH),
	("validation", valPaths, VAL_PATH),
	("testing", testPaths, TEST_PATH)
]

# running loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
	# shows which data split we are creating
	print("[INFO] building '{}' split".format(dType))

	# if output base output directoey does not exist, create gardeu
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	# loop over the input image paths
	for inputPath in imagePaths:
		# extract the class label ("0" for "negative" and "1" for "positive")
		filename = inputPath.split(os.path.sep)[-1]
		label = filename[-5:-4]

		# build the path to the label directory
		labelPath = os.path.sep.join([baseOutput, label])

		# if label output directory does not exist, the  create gardeu
		if not os.path.exists(labelPath):
			print("[INFO] 'creating {}' directory".format(labelPath))
			os.makedirs(labelPath)

		# constructing the path to the destinstion image and then copy the image itselfg
		p = os.path.sep.join([labelPath, filename])
		shutil.copy2(inputPath, p)
        
###############CANCERNET model build##################
##importing necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class CancerNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initializing the model along with the input shape to be channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		# CONV => layer1
		model.add(SeparableConv2D(256, (3, 3), padding="same",input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# CONV => layer2
		model.add(SeparableConv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
        #model.add(MaxPooling2D(pool_size=(2, 2))) 
        
        # CONV =>layer3
		model.add(SeparableConv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.25))
		
        # CONV =>layer4
		model.add(SeparableConv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# fully connected layer
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model



#####################TRAIN model###################################
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import numpy as np
import argparse
import os

# constructing the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initializing our number of epochs, initial learning rate, and batchsize
NUM_EPOCHS = 20
INIT_LR = 1e-5
BS = 8

# calcualting the total number of image paths in train,val,test
trainPaths = list(paths.list_images(TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

# account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# initializing the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=25,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

# initializing the validation  data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initializing the training generator
trainGen = trainAug.flow_from_directory(
	TRAIN_PATH,
	class_mode="categorical",
	target_size=(460,700),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initializing the validation generator
valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode="categorical",
	target_size=(460,700),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initializing the testing generator
testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode="categorical",
	target_size=(460,700),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initializing our CancerNet model to compile it
model = CancerNet.build(width=700, height=460, depth=3,
	classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
#model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
	#metrics=["accuracy"])
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#since our model is already trained now we load the saved model
from keras.models import load_model
model = load_model('my_model.h5')

# load test data for prediction
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BS) + 1)

# for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

#to show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

# computing the confusion matrix and and use it to calculate the accuracy, sensitivity, and specificity
cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# visualize the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("specificity: {:.4f}".format(specificity))






    


