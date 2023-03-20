import numpy as np
np.random.seed(100)
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
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

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,)

        model = Sequential()
        model.add(Conv1D(128, 5, activation='relu', input_shape=(self.sentence_length, self.dimensions)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation=activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def fit(self, train_txt):
        # read in data
        df_train = pd.read_csv(train_txt, header=None, skiprows=1)
        df_train = shuffle(df_train, random_state=100)

        # get number of classes and create label mapping
        unique_labels = sorted(df_train[0].unique())
        self.num_classes = len(unique_labels)      
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        # initialize x and y matrices
        num_lines = len(df_train)
        x_matrix = np.zeros((num_lines, self.sentence_length, self.dimensions))
        y_matrix = np.zeros((num_lines, self.num_classes))

        # insert values
        for i, row in df_train.iterrows():
            label = int(row[0])
            label_idx = self.label_mapping[label]
            sentence = row[1]
            words = sentence.split()[:self.sentence_length]
            for j, word in enumerate(words):
                if word in self.w2v:
                    x_matrix[i, j, :] = self.w2v[word]
            y_matrix[i][label_idx] = 1.0

        # train model
        self.model = self.build_cnn()
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks = [EarlyStopping(monitor='val_loss', patience=3), tensorboard_callback]
        self.model.fit(x_matrix, y_matrix, epochs=100000, callbacks=callbacks, validation_split=0.1,
                       batch_size=4, shuffle=True, verbose=0)
        
        # clean memory
        del x_matrix, y_matrix, df_train
        gc.collect()

    def predict(self, test_txt):
        # read in data
        df_test = pd.read_csv(test_txt, header=None, skiprows=1)
        # initialize x and y matrices
        num_lines = len(df_test)
        x_matrix = np.zeros((num_lines, self.sentence_length, self.dimensions))
        y_matrix = np.zeros((num_lines, self.num_classes))
        
        # insert values
        for i, row in df_test.iterrows():
            label = row[0]
            label_idx = self.label_mapping[label]
            sentence = row[1]
            words = sentence.split()[:self.sentence_length]
            for j, word in enumerate(words):
                if word in self.w2v:
                    x_matrix[i, j, :] = self.w2v[word]
            y_matrix[i][label_idx] = 1.0

        # predict labels
        y_pred = self.model.predict(x_matrix)
        test_y_cat = np.argmax(y_matrix, axis=1)
        y_pred_cat = np.argmax(y_pred, axis=1)
        acc = accuracy_score(test_y_cat, y_pred_cat)

        # clean memory
        del x_matrix, y_matrix, df_test

        return acc

if __name__=='__main__':

    # hyperparameters
    sentence_length = 128
    dimensions = 300    
    w2v_file = 'w2v.pkl'
    train_file = './data/original/bbc/train.csv'
    test_file = './data/original/bbc/test.csv'
    # train model
    model = CNN(sentence_length, dimensions, w2v_file)
    model.fit(train_file)

    # test model
    acc = model.predict(test_file)
    print('Test accuracy: {}'.format(acc))

