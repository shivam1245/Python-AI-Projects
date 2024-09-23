Here's a `README.md` file for your image classification script:

---

# Image Classification using CIFAR-10 Dataset

This project demonstrates an image classification model built using the CIFAR-10 dataset. The model is a Convolutional Neural Network (CNN) designed to classify images into 10 different classes, including planes, cars, birds, cats, and more. The script trains the model, evaluates it, and tests it on a custom image downloaded from the internet.

## Requirements

To run this project, you'll need to have the following software installed:

- Python 3.x
- Virtual Environment (optional, but recommended)
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## Environment Setup

It is recommended to create a virtual environment to manage dependencies:

1. **Create and activate virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install required dependencies:**

    ```bash
    pip install tensorflow opencv-python-headless matplotlib numpy
    ```

## Project Files

- **`image_classifier.py`**: Main script that contains code for building, training, and testing the image classifier model.
- **`plane.jpg`**: Sample image downloaded from the browser, which has been resized to 32x32 pixels to match the CIFAR-10 input size.

## Dataset

The project uses the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

- **Training images**: 40,000 images are used for training.
- **Testing images**: 8,000 images are used for testing.

## Model Structure

The CNN model consists of the following layers:

1. Conv2D (32 filters, kernel size 3x3, ReLU activation)
2. MaxPooling2D (pool size 2x2)
3. Conv2D (64 filters, kernel size 3x3, ReLU activation)
4. MaxPooling2D (pool size 2x2)
5. Conv2D (64 filters, kernel size 3x3, ReLU activation)
6. Flatten
7. Dense (64 units, ReLU activation)
8. Dense (10 units, softmax activation)

The model is compiled using the Adam optimizer, with sparse categorical crossentropy as the loss function, and accuracy as the evaluation metric.

## Training

The model is trained for 10 epochs using a training set of 40,000 images, and validated on a test set of 8,000 images.

```python
model.fit(training_images, training_lables, epochs=10, validation_data=(testing_images, testing_lables))
```

## Evaluation

After training, the model is evaluated on the test set:

```python
loss, accuracy = model.evaluate(testing_images, testing_lables)
print(f'Loss: {loss}')
print(f"Accuracy: {accuracy}")
```

## Testing on a Custom Image

The script allows you to load a custom image (`plane.jpg`) for testing. This image must be resized to 32x32 pixels before being used in the model.

To load the custom image and make a prediction:

1. Download an image (e.g., `plane.jpg`) from the internet.
2. Resize the image to 32x32 pixels (this can be done using any image editing tool).
3. Place the image in the project directory.

The script then loads and predicts the class of the image:

```python
img = cv.imread('plane.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_name[index]}')
```

## Model Saving and Loading

After training, the model is saved to a file (`image_classifier.h5`) for future use. The model can be loaded later for inference on new images.

```python
model.save("image_classifier.h5")
model = models.load_model('image_classifier.h5')
```

## Results

The model achieves a reasonable accuracy on the CIFAR-10 dataset, and it can classify custom images resized to 32x32 pixels.
