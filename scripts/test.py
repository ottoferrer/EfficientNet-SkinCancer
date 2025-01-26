import tensorflow as tf
import cv2
import albumentations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from keras.applications import EfficientNetB4


print("TensorFlow version:", tf.__version__)
print("OpenCV version:", cv2.__version__)
print("Albumentations version:", albumentations.__version__)
print("NumPy version:", np.__version__)
model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print(model.summary())