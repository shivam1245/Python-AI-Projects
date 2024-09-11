import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


# Data collection

# Use shakespear's poem file provided by google apis in following link
# Here tensorflow directly handle the file.
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb')\
    .read().decode(encoding='utf-8').lower()


text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3


# Data PreProcessing
'''
Preprocessing part can be removed after training the model.
'''
sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH,
              len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences),
              len(characters)), dtype=np.bool_)

for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1


# Creating LSTM Model
'''
model creation code can be removed after training once.
'''
model = Sequential()
model.add(LSTM(128,
               input_shape=(SEQ_LENGTH,
                            len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.01))

model.fit(x, y, batch_size=256, epochs=4)
model.save('textgenerator.h5')  # save the model

model = tf.keras.models.load_model('textgenerator.h5')  # Read saved model

