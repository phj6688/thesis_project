import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_addons as tfa
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import os
import json
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(100)
random.seed(100)
tf.random.set_seed(100)


class CNN:
    def __init__(self, dims, w2v_path, max_seq_len=20, batch_size=128, epochs=20, chunk_size=1000):
        self.dims = dims
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.epochs = epochs
        with open(w2v_path, 'rb') as f:
            self.w2v = pickle.load(f)
        self.model = None
        self.n_classes = None
        self.history = None
        self.metrics = None
        self.callbacks = None

    def build_cnn(self):
        if self.n_classes > 2:
            loss = 'categorical_crossentropy'
            activation = 'softmax'
        else:
            loss = 'binary_crossentropy'
            activation = 'sigmoid'

        input_layer = layers.Input(shape=(self.max_seq_len, 300))
        conv1_1 = layers.Conv1D(128, 4, activation='relu', padding='same')(input_layer)
        conv1_2 = layers.Conv1D(128, 5, activation='relu', padding='same')(conv1_1)
        #conv1_3 = layers.Conv1D(128, 5, activation='relu', padding='same')(conv1_2)
        conv_out = layers.Concatenate(axis=1)([conv1_1, conv1_2])

        dropout_rate = 0.5
        dropout_out1 = layers.Dropout(dropout_rate)(conv_out)

        pool_out = layers.MaxPool1D(pool_size=self.max_seq_len, padding='valid')(dropout_out1)
        flatten_out = layers.Flatten()(pool_out)
        dropout_out2 = layers.Dropout(dropout_rate)(flatten_out)
        dense_out = layers.Dense(self.n_classes, activation=activation, kernel_regularizer=regularizers.L2(0.001))(dropout_out2)
        
        self.metrics = [tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score'), 'accuracy']
        cnn_model = Model(inputs=input_layer, outputs=dense_out)
        cnn_model.compile(optimizer='adam', loss=loss, metrics=self.metrics)
        #cnn_model.summary()
        self.model = cnn_model

    def prepare_dataset(self, df):
        def generator():
            for _, row in df.iterrows():
                label = row[0]
                sentence = row[1]
                x = np.zeros((self.max_seq_len, 300))
                y = np.zeros(self.n_classes)

                if isinstance(sentence, str):
                    words = sentence.split()[:self.max_seq_len]
                    for k, word in enumerate(words):
                        if word in self.w2v:
                            x[k, :] = self.w2v[word]
                y[label] = 1.0
                yield x, y

        dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(self.max_seq_len, 300), dtype=tf.float32),
            tf.TensorSpec(shape=(self.n_classes,), dtype=tf.float32)
        )
    )
        return dataset


    def insert_values(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self.n_classes = train_df['class'].nunique()
        unique_classes = train_df['class'].unique()
        labels_map = dict(zip(unique_classes, range(self.n_classes)))

        train_df['class'] = train_df['class'].map(labels_map)
        test_df['class'] = test_df['class'].map(labels_map)

        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=100)
        print(f'Train size: {len(train_df)}\nValidation size: {len(val_df)}\nTest size: {len(test_df)}')

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=100).reset_index(drop=True)

        train_dataset = self.prepare_dataset(train_df).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = self.prepare_dataset(test_df).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = self.prepare_dataset(val_df).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset, test_dataset, val_dataset, self.n_classes

    def fit(self, train_dataset, val_dataset):
        self.metrics = [tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score'), 'accuracy']
        self.build_cnn()
        self.history = self.model.fit(train_dataset, epochs=self.epochs, validation_data=val_dataset, callbacks=self.callbacks, verbose=1)
        return self.history

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset, return_dict=True)

    def run_n_times(self, train_dataset, test_dataset, val_dataset, dataset_name, n=3):

        log_dir = f"logs/fit/cnn/{dataset_name}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        decay_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001 ,min_lr=0.00001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
        self.callbacks = [tensorboard_callback, decay_rate, early_stopping]

        hist_dict = {}
        res_dict = {}
        best_val_loss = float('inf')
        for i in range(n):
            print(f'Run {i+1} of {n}')
            try:
                self.fit(train_dataset, val_dataset)  # Updated to use train_dataset and val_dataset
            except tf.errors.ResourceExhaustedError:
                K.clear_session()
                self.model = None
                self.build_cnn()
                continue
            res = self.evaluate(test_dataset)  # Updated to use test_dataset
            res_dict[i+1] = res
            if self.history.history['val_loss'][-1] < best_val_loss:
                best_val_loss = self.history.history['val_loss'][-1]
                self.model.save(f"models/cnn/full/{dataset_name}_best_model.h5")
            self.model.set_weights([np.zeros(w.shape) for w in self.model.get_weights()])

        avg_dict = {metric: round(sum(values[metric] for values in res_dict.values()) / len(res_dict), 4) for metric in res_dict[1].keys()}

        # Save the average results to disk
        os.makedirs("results/original/cnn", exist_ok=True)
        with open(f"results/original/cnn/full/{dataset_name}_full_results.txt", "w") as f:
            for key, value in avg_dict.items():
                f.write(f"{key}: {value}\n")

        K.clear_session()

        return hist_dict, res_dict, avg_dict