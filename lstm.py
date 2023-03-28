import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_addons as tfa
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(100)
import random
random.seed(100)
tf.random.set_seed(100)



class LSTM:
    def __init__(self,dims,w2v_path,max_seq_len=20,batch_size=128,epochs=20,chunk_size=1000):
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
        self.metrics = None #[tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score'), 'accuracy']        
        self.callbacks =None
        

    def build_lstm(self):
        if self.n_classes > 2:
            loss = 'categorical_crossentropy'
            activation = 'softmax'
        else:
            loss = 'binary_crossentropy'
            activation = 'sigmoid'

        input_layer = layers.Input(shape=(self.max_seq_len, 300))
        lstm_1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(input_layer)
        dropout_rate = 0.5
        dropout_out1 = layers.Dropout(dropout_rate)(lstm_1)
        lstm_2 = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(dropout_out1)
        dropout_out2 = layers.Dropout(dropout_rate)(lstm_2)
        dense_1 = layers.Dense(20, activation='relu')(dropout_out2)
        dense_out = layers.Dense(self.n_classes, activation=activation, kernel_regularizer=regularizers.L2(0.001))(dense_1)
        
        self.metrics = [tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score'), 'accuracy']
        lstm_model = Model(inputs=input_layer, outputs=dense_out)
        lstm_model.compile(optimizer='adam', loss=loss, metrics=self.metrics)
        #lstm_model.summary()
        self.model = lstm_model
        
    def insert_values(self,train_path,test_path):    
        def insert(df):
            
            # initialize x self.and y matrices
            num_lines = len(df)
            self.n_classes = df['class'].nunique()          
            x_matrix = np.zeros((num_lines, self.max_seq_len ,300))
            y_matrix = np.zeros((num_lines, self.n_classes))


            # insert values
            for i in range(0, num_lines, self.chunk_size):
                df_batch = df.iloc[i:i+self.chunk_size]
                batch_size = len(df_batch)
                x_batch = np.zeros((batch_size, self.max_seq_len, 300))
                y_batch = np.zeros((batch_size, self.n_classes))

                for j, row in df_batch.iterrows():
                    label = row[0]
                    sentence = row[1]
                    if isinstance(sentence, str):
                        words = sentence.split()[:self.max_seq_len]
                        for k, word in enumerate(words):
                            if word in self.w2v:
                                x_batch[j-i, k, :] = self.w2v[word]
                    else:
                        continue        
                    y_batch[j-i,label] = 1.0

                x_matrix[i:i+batch_size] = x_batch
                y_matrix[i:i+batch_size] = y_batch

            return x_matrix,y_matrix
    
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
        val_df = val_df.sample(frac=1,random_state=100).reset_index(drop=True)

        train_x, train_y = insert(train_df)
        test_x, test_y = insert(test_df)
        val_x, val_y = insert(val_df)

        return train_x, train_y, test_x, test_y, val_x, val_y, self.n_classes          

    def fit(self,train_x, train_y,  val_x, val_y):
        self.metrics = [tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score'), 'accuracy']
        self.build_lstm()  
        self.history = self.model.fit(train_x, train_y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(val_x, val_y), callbacks=self.callbacks, verbose=0)
        return self.history
    def evaluate(self,test_x, test_y):
        return self.model.evaluate(test_x, test_y,return_dict=True)



    def run_n_times(self,train_x, train_y, test_x, test_y, val_x, val_y, dataset_name, n=3):
        
        log_dir = f"logs/fit/{dataset_name}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
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
                self.fit(train_x, train_y, val_x, val_y)
            except tf.errors.ResourceExhaustedError:
                K.clear_session()
                self.model = None
                self.build_lstm()
                continue
            res = self.evaluate(test_x, test_y)
            res_dict[i+1] = res
            if self.history.history['val_loss'][-1] < best_val_loss:
                best_val_loss = self.history.history['val_loss'][-1]
                self.model.save(f"models/lstm/{dataset_name}_best_model.h5")
            self.model.set_weights([np.zeros(w.shape) for w in self.model.get_weights()])
        
        avg_dict = {metric: round(sum(values[metric] for values in res_dict.values()) / len(res_dict), 4)  for metric in res_dict[1].keys()}
        
        # Save the average results to disk
        os.makedirs("results/lstm", exist_ok=True)
        with open(f"results/lstm/{dataset_name}_avg_results.txt", "w") as f:
            for key, value in avg_dict.items():
                f.write(f"{key}: {value}\n")
        
        K.clear_session()
        
        return hist_dict, res_dict, avg_dict
