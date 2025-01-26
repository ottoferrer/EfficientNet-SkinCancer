import tensorflow as tf
import cv2
import albumentations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score


print("TensorFlow version:", tf.__version__)
print("OpenCV version:", cv2.__version__)
print("Albumentations version:", albumentations.__version__)
print("NumPy version:", np.__version__)