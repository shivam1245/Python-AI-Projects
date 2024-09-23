import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


# Collect/Read the data/images from keras dataset
(training_images, training_lables), (testing_images, testing_lables) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

# class_name = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Frog', 'Dog', 'Horse', 'Ship', 'Truck', 'Motorcycle', 'Lion', 'Tiger']
class_name = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_name[training_lables[i][0]])

# plt.show()


# Data preparation, I'm using first 20k data for training as a training set,
# 4k data as a testing set to save training time.

training_images = training_images[:40000]
training_lables = training_lables[:40000]
testing_images = testing_images[:8000]
testing_lables = testing_lables[:8000]


# Building Training model
# This model creating script  can be removed once training is completed

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))  # helps in classifying the image's parts or key objects (legs, wings, etc)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_lables, epochs=10, validation_data=(testing_images, testing_lables))

loss, accuracy = model.evaluate(testing_images, testing_lables)
print(f'Loss: {loss}')
print(f"Accurcy: {accuracy}")

model.save("image_classifier.h5")


model = models.load_model('image_classifier.h5')

img = cv.imread('plane.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_name[index]}')
