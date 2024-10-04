Here’s a detailed `README.md` for your "GeneratePoeticTextWithRNN" project:

---

# GeneratePoeticTextWithRNN

This project generates poetic text using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. The model is trained on a portion of Shakespeare’s works, and after training, it can generate new text based on learned patterns. This implementation uses TensorFlow and Keras.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Text Generation](#text-generation)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

## Overview

This project uses an LSTM-based neural network to learn the patterns and structure of text from Shakespeare's works. Once trained, the model generates poetic text based on a sequence of characters fed into it. The user can control the randomness of the generated text by adjusting the "temperature" parameter, which influences how creative or conservative the text generation is.

### Key Features:
- Uses LSTM layers to process sequences of characters.
- Trains on Shakespeare’s text to generate poetic text.
- Allows temperature tuning to adjust text creativity.
- Saves the trained model for future reuse.

## Project Structure

```
.
├── textgenerator.h5            # Trained model saved for future use
├── main.py                     # Main script for data preprocessing, training, and text generation
└── README.md                   # Project documentation
```

## Requirements

To run this project, you will need the following libraries:

- TensorFlow
- NumPy

You can install the required libraries with the following command:

```bash
pip install tensorflow numpy
```

## Data Collection

The dataset used for training is a text file of Shakespeare’s works, provided by TensorFlow’s API. The file is automatically downloaded when running the script:

```python
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
```

The text is preprocessed by:
- Lowercasing the characters.
- Slicing a portion of the text (between the 300,000th and 800,000th character) for training.
- Mapping each unique character to an index for easy lookup.

### Character Mapping:

```python
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))
```

## Model Training

An LSTM-based neural network is built to predict the next character in a sequence of 40 characters.

### Model Structure:
- **LSTM Layer:** Learns long-term dependencies between characters in a sequence.
- **Dense Layer:** Produces a probability distribution over all possible characters.
- **Activation Layer (Softmax):** Normalizes the output to represent character probabilities.

```python
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
```

The model is compiled with categorical cross-entropy as the loss function and RMSprop as the optimizer:

```python
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
```

The training is done over 4 epochs with a batch size of 256:

```python
model.fit(x, y, batch_size=256, epochs=4)
```

After training, the model is saved as `textgenerator.h5` for future use.

## Text Generation

The model generates text by sampling the probability distribution of predicted characters. The sampling function (`sample()`) uses a temperature parameter that controls how conservative or creative the predictions are:

- **Lower temperatures** (e.g., 0.2) lead to more predictable and repetitive text.
- **Higher temperatures** (e.g., 0.7) introduce more randomness and creativity.

```python
def generate_text(length, temperature):
    # Randomly choose a starting index from the text
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated = sentence

    # Predict next character for 'length' number of characters
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    return generated
```

## Usage

To run the project and generate text:

1. **Download the dataset** (done automatically by the script).
2. **Train the model** (done within the script unless the pre-trained model is used).
3. **Generate text** based on a chosen length and temperature.

Example of generating text:

```bash
python main.py
```

Output:

```
generating text.....
<generated_text_sample>
```

You can adjust the temperature to control the creativity:

```
print(generate_text(300, 0.2))  # More repetitive text
print(generate_text(300, 0.7))  # More creative text
```

## Customization

- **Text Input**: You can modify the text source by replacing the file URL with your own dataset.
- **Sequence Length**: The sequence length (`SEQ_LENGTH = 40`) can be adjusted to change how much context the model uses for its predictions.
- **Temperature**: Adjust the `temperature` parameter to control the randomness of the generated text.
