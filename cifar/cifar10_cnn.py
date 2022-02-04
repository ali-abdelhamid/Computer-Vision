# Simple CNN model for the CIFAR-10 Dataset
import numpy
import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_data_format('channels_first')

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras.utils import img_to_array
from keras.models import Model
from keras.preprocessing import image

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 2
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose=2)

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

#get layer outputs
layer_outputs = [layer.output for layer in model.layers]

#get feature maps
feature_map_model = Model(model.input, layer_outputs)

#get image
image_path= r"/home/ali/Desktop/subimg_0.jpg"
img = image.load_img(image_path, target_size=(32, 32))

#prepare image to feed to feature map
inputt = img_to_array(img)                           
inputt = inputt.reshape((1,) + inputt.shape)                   
inputt /= 255.0

for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue

	filters, biases=layer.get_weights()
	f_min, f_max = filters.min(), filters.max()
	filters = (filters - f_min) / (f_max - f_min)
	
	# plot first few filters
	n_filters, ix = 16, 1
	for i in range(n_filters):
		# get the filter
		f = filters[:, :, :, i]
		# plot each channel separately
		for j in range(3):
			# specify subplot and turn of axis
			ax = plt.subplot(n_filters, 3, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			plt.imshow(f[:, :, j], cmap='gray')
			ix += 1
	# show the figure
	plt.show()
	#print(layer.name, filters.shape)
	



# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
