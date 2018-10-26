# I wrote this by looking at the example code
# 3-layer DNN i.e., 1-input layer, 1-hidden layer, 1-output layer
# There is weights defined for (input layer, hidden layer) and (hidden layer, input layer)
# Output layer is redundant layer. It is just tapping of hidden layer outputs

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils.vis_utils import plot_model

# mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

input_num_units = 28*28
hidden_num_units = 500
output_num_units = 10

seed = 128
rng = np.random.RandomState(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], input_num_units).astype('float32')
X_test = X_test.reshape(X_test.shape[0], input_num_units).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(500, activation='relu', input_dim=784))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
print("Training Complete")
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("DNN Error: %.2f%%" % (100-scores[1]*100))

# Visualize the model
plot_model(model, to_file='mnist_3layer.png', show_shapes=True, show_layer_names=True)