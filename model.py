import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import random
from PIL import Image
from google.colab import files
from tensorboardcolab import *

#importing images and replacing the path
imagedir = "/content/data/IMG"
df = pd.read_csv("/content/data/driving_log.csv",header=None, names=["center", "left", "right", "steering", "acceleration", "breaking", "speed" ])
df.replace("/Users/adriendodinet/Downloads/beta_simulator_mac", "/content/data", regex=True, inplace=True)


#creating a list of image path with the 3 camera angles
#also creating a list of coresponding steering angles, I add 0.1 to the left images
#and add 0.1 to the right images. It helps recentering the car when it drifts
def load_img(imagedir, df):
  image_path = []
  steering = []
  for i in range(len(df)):
    indexed_data = df.iloc[i]
    for col in range(0,3):
      image = indexed_data[col]
      image_path.append(os.path.join(imagedir, image.strip()))
      if col == 1:
        adjusted_steering = indexed_data[3] + 0.1
        if adjusted_steering > 0.5:
          adjusted_steering = 0.5
      elif col == 2:
        adjusted_steering = indexed_data[3] - 0.1
        if adjusted_steering < -0.5:
          adjusted_steering = -0.5
      else:
        adjusted_steering = indexed_data[3]       
      steering.append(adjusted_steering)
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings
  
image_paths, steerings = load_img(imagedir, df)


#cropping, using the YUV color space, blurring and normalizing the images pixels.
def img_preprocess(img):
  img = mpimg.imread(img)
  img = img[60:140, :,: ]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = img/255
  return img


#splitting in train and test set. Not going the train/val/test way because I do not have much data.
X_train, X_test, y_train, y_test = train_test_split(image_paths, steerings,test_size = 0.2, random_state = 42)

X_train = np.array(list(map(img_preprocess, X_train)))
X_test = np.array(list(map(img_preprocess, X_test)))


#Model from the nvidia paper.
#5 convolutionnal layers followed by a dropout layer.
#Dropout is used twice to avoid overfitting
#4 Fully Connected layer ending with relu activation
#Using Adam optimizer
#Loss is mean squared error since it's not a classification case but a regression case.
def nvidia_model():
  model = Sequential()
  model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape = (80, 320, 3), activation='relu'))
  model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
  model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1))
  optimizer = Adam(lr = 1e-4)
  model.compile(loss = 'mse', optimizer=optimizer)
  return model

model = nvidia_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), batch_size = 64, verbose = 1, shuffle = 1)

#Printing the history to make sure there is no overfitting
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training data', 'validation data'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

model.save('/content/gdrive/My Drive/Data Science/Dataset udacity/model.h5')




