![Diagram](Assets/Hate%20Speech%20Detection%20Algorithm.png)


# Hate Speech Detection Model

This repository contains a deep learning model designed to classify text, specifically tweets, into three categories: hate speech, offensive language, and neutral text.

## Overview

The model is built using the Keras library with TensorFlow as the backend. It employs a convolutional neural network (CNN) architecture to process tokenized and embedded sequences of text.

## Features

- Data preprocessing including tokenization, padding, and removal of stopwords and special characters.
- Convolutional neural network model with embedding, convolutional, pooling, dense, and dropout layers.
- Visualization of training and validation accuracy and loss.
- Interactive text classification function for real-time predictions.

## Dependencies

- numpy
- matplotlib
- pandas
- logging
- re
- keras
- nltk
- sklearn
- tensorflow

## How to Run

1. Ensure you have all the dependencies installed.
2. Download the dataset `labeled_data.csv` and place it in the same directory as the script.
3. Run the script. After training, the model's performance will be evaluated on both the training and testing data.
4. Use the interactive classification function to input text and get predictions from the trained model.

## Model Architecture

- **Embedding Layer**: Converts tokenized sequences into dense vectors.
- **Conv1D Layer**: Detects patterns or motifs in sequences of words or characters.
- **GlobalMaxPooling1D Layer**: Reduces the spatial dimensions, capturing the most important features.
- **Dense Layers**: Learn complex representations from the features.
- **Dropout Layer**: Prevents overfitting by randomly setting a fraction of the input units to 0 during training.
- **Output Layer**: Uses softmax activation for multi-class classification.

## Data Preprocessing

- Removal of special characters, retweet indicators, and stopwords.
- Tokenization of tweets.
- Conversion of tokenized sequences to padded sequences of fixed length.
