import numpy as np
np.random.seed(100)
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_addons as tfa
import tensorflow as tf
tf.random.set_seed(100)
from datetime import datetime
import random
random.seed(100)
import gc
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = ''




class CNN:
    
    def __init__(self, sentence_length, dimensions, w2v_path):
        self.sentence_length = sentence_length
        with open(w2v_path, 'rb') as f:            
            self.w2v = pickle.load(f)
        self.dimensions = dimensions        
        self.num_classes = None        
        self.model = None
        self.label_mapping = None      

    def build_cnn(self):
        if self.num_classes > 2:
            loss = 'categorical_crossentropy'
            activation = 'softmax'
        else:
            loss = 'binary_crossentropy'
            activation = 'sigmoid'

        metrics = [
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tfa.metrics.F1Score(self.num_classes, average='weighted', name='f1_score')]

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)


        model = Sequential()
        model.add(Conv1D(256, 5, activation='relu', input_shape=(self.sentence_length, self.dimensions)))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation=activation))
        # model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer='he_normal'))
        # model.add(Dropout(0.5))
        # model.add(Dense(self.num_classes, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer='he_normal'))
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def fit(self, train_txt):
    # Read the number of unique labels and create the label_mapping
        df_labels = pd.read_csv(train_txt, header=None, usecols=[0], skiprows=1)
        unique_labels = sorted(df_labels[0].unique())
        self.num_classes = len(unique_labels)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        # Initialize the model
        self.model = self.build_cnn()

        # Train the model on each chunk
        chunksize = 5000  # Adjust this value according to your memory availability
        num_epochs = 100
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            chunk_iter = pd.read_csv(train_txt, header=None, skiprows=1, chunksize=chunksize)
            for chunk in chunk_iter:
                chunk = shuffle(chunk, random_state=100)
                x_matrix, y_matrix = self.process_chunk(chunk)
                self.model.train_on_batch(x_matrix, y_matrix)

    # Clean memory
        gc.collect()


    def process_chunk(self, chunk):
        num_lines = len(chunk)
        x_matrix = np.zeros((num_lines, self.sentence_length, self.dimensions))
        y_matrix = np.zeros((num_lines, self.num_classes))

        for index, (_, row) in enumerate(chunk.iterrows()):
            label = row[0]
            label_idx = self.label_mapping[label]
            sentence = row[1]
            if isinstance(sentence, str):
                words = sentence.split()[:self.sentence_length]
                for j, word in enumerate(words):
                    if word in self.w2v:
                        x_matrix[index, j, :] = self.w2v[word]
            else:
                print(f"Warning: Skipping row {index} due to invalid sentence data: {sentence}")
            y_matrix[index][label_idx] = 1.0


        return x_matrix, y_matrix   


    def predict(self, test_txt):
        chunksize = 5000  # Adjust this value according to your memory availability
        chunk_iter = pd.read_csv(test_txt, header=None, skiprows=1, chunksize=chunksize)

        total_samples = 0
        correct_predictions = 0

        for chunk in chunk_iter:
            x_matrix, y_matrix = self.process_chunk(chunk)
            y_pred = self.model.predict(x_matrix)

            if self.num_classes > 2:
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_matrix, axis=1)
            else:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                y_true_classes = y_matrix.flatten()

            correct_predictions += np.sum(y_pred_classes == y_true_classes)
            total_samples += len(y_matrix)

        accuracy = correct_predictions / total_samples
        print(f'Test accuracy: {accuracy}')

        # Clean memory
        gc.collect()
        return accuracy




if __name__=='__main__':

    # hyperparameters
    sentence_length = 128
    dimensions = 300    
    w2v_file = 'w2v.pkl'
    train_file = './data/original/cardio/train.csv'
    test_file = './data/original/cardio/test.csv'
    # train model
    model = CNN(sentence_length, dimensions, w2v_file)
    model.fit(train_file)
    os.system('clear')
    # test model
    acc = model.predict(test_file)
    print('Test accuracy: {}'.format(acc))

