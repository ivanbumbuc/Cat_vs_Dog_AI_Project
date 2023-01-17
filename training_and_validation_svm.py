import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set = datagen.flow_from_directory( "../train",target_size=(64, 64),batch_size=32,class_mode="binary")

datagen1 = ImageDataGenerator(rescale=1./255)

test_set = datagen1.flow_from_directory("../test",target_size=(64, 64),batch_size=32,class_mode="binary" )