# Baseline MLP for MNIST dataset

import numpy
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

from keras.initializers import he_normal
from keras.initializers import he_uniform
from keras.initializers import random_normal
from keras.initializers import orthogonal
from keras.layers import LayerNormalization
from tensorflow.keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.regularizers import l2

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dropout(0.3))
	model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu', kernel_initializer=orthogonal(seed=None)))
	model.add(LayerNormalization())
	model.add(Dropout(0.5))
	
	#second hidden layer
	model.add(Dense(600, activation ='relu', kernel_initializer=orthogonal(seed=None)))
	model.add(LayerNormalization())
	model.add(Dropout(0.5))
	
	#second hidden layer
	model.add(Dense(300, activation ='relu', kernel_initializer=orthogonal(seed=None)))
	model.add(LayerNormalization())
	model.add(Dropout(0.5))
	
	#second hidden layer
	model.add(Dense(200, activation ='relu', kernel_initializer=orthogonal(seed=None)))
	model.add(LayerNormalization())
	model.add(Dropout(0.3))
	
	
	
	
	#output layer
	model.add(Dense(num_classes, activation='softmax'))
	
	#define learning rate and decay
	optimizer = Adam(learning_rate = 0.001)
	
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	
	return model

# build the model
model = baseline_model()

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

#summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
