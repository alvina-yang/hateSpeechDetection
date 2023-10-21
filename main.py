import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import re
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Plotting
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

def classify_text(model, tokenizer, text):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    predicted_probs = model.predict(padded_sequence)[0]
    labels = ['Hate Speech', 'Offensive', 'Neither']
    predicted_label = labels[np.argmax(predicted_probs)]
    confidence = np.max(predicted_probs)
    return predicted_label, confidence


logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# loading datasets
dataset = pd.read_csv('labeled_data.csv')

# accessing the columns
counts = dataset['count']
hate_speech = dataset['hate_speech']
offensive_language = dataset['offensive_language']
neither = dataset['neither']
class_labels = dataset['class']
tweets = dataset['tweet']

stop_words = set(stopwords.words('english'))

# Preprocessing by removing special characters, RT, and stop words
tweets = tweets.apply(lambda x: ' '.join([word for word in re.sub(r'[^a-zA-Z0-9\s]', '', x).split()
                                          if word not in stop_words]))

# Tokenizing the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)

# Converting the text into sequences
sequences = tokenizer.texts_to_sequences(tweets)

max_sequence_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(class_labels)
categorical_labels = to_categorical(encoded_labels, num_classes=3)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
train_data, test_data, train_labels, test_labels = train_test_split(padded_sequences, categorical_labels,
                                                                    test_size=0.2, random_state=42)
_, _, train_offensive_language, test_offensive_language = train_test_split(padded_sequences, offensive_language,
                                                                          test_size=0.2, random_state=42)
_, _, train_neither, test_neither = train_test_split(padded_sequences, neither,
                                                    test_size=0.2, random_state=42)
_, _, train_hate_speech, test_hate_speech = train_test_split(padded_sequences, hate_speech,
                                                    test_size=0.2, random_state=42)

train_total_votes = train_hate_speech + train_offensive_language + train_neither
test_total_votes = test_hate_speech + test_offensive_language + test_neither

train_hate_speech_frac = train_hate_speech / train_total_votes
train_offensive_language_frac = train_offensive_language / train_total_votes
train_neither_frac = train_neither / train_total_votes

test_hate_speech_frac = test_hate_speech / test_total_votes
test_offensive_language_frac = test_offensive_language / test_total_votes
test_neither_frac = test_neither / test_total_votes

# Model
model = Sequential(
    [
        Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
        Conv1D(256, 5, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
        Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.05)),
        Dropout(0.5),
        Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.001))
    ]
)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0013), metrics=['accuracy'])
model.summary()

print("Train Data Shape:", train_data.shape, flush=True)
print("Test Data Shape:", test_data.shape, flush=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)


history = model.fit(train_data,
                    np.column_stack((train_hate_speech_frac, train_offensive_language_frac,
                                     train_neither_frac)),
                    epochs=10, batch_size=15, validation_split=0.3,  callbacks=[early_stopping, checkpoint])

train_loss, train_accuracy = model.evaluate(train_data,
                                            np.column_stack((train_hate_speech_frac, train_offensive_language_frac,
                                                             train_neither_frac)))
print("Train Accuracy:", train_accuracy)

test_loss, test_accuracy = model.evaluate(test_data,
                                          np.column_stack((test_hate_speech_frac, test_offensive_language_frac,
                                                           test_neither_frac)))
print("Test Accuracy:", test_accuracy)

plot_history(history)


# Classification function
while True:
    text = input("Enter text (or 'q' to quit): ")
    if text == 'q':
        break
    label, confidence = classify_text(model, tokenizer, text)
    print(f"Predicted Label: {label}, Confidence: {confidence}\n")