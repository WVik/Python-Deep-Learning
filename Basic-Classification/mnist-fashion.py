import tensorflow as tf
from tensorflow import keras

import numpy as numpy
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

#print(trainX[0], trainY[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Information about the dataset:")
print("Shape: " + str(trainX.shape))

trainX = trainX/255.0
testX = testX/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainX,trainY, epochs=5)

test_loss, test_acc = model.evaluate(testX, testY)
print('Test accuracy:', test_acc)
