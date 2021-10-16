from keras.utils import to_categorical
from sklearn.datasets import load_boston, load_diabetes
from keras.datasets import mnist
from scipy.spatial import distance_matrix
from keras.losses import categorical_crossentropy
from keras.layers import Lambda
from keras import optimizers
from cleverhans.attacks import FastGradientMethod, DeepFool
from cleverhans.utils_keras import KerasModelWrapper
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from keras.models import load_model
from keras.applications import MobileNet
from keras.layers import Dense,GlobalAveragePooling2D

import pickle
import os
import sys
import argparse
import gc
import keras.backend as K
import numpy as np
