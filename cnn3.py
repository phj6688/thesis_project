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

class CNN:
    def __init__(self,dims,w2v_path,max_seq_len=20,batch_size=128,epochs=20):
        self.dims = dims
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        with open(w2v_path, 'rb') as f:            
            self.w2v = pickle.load(f)
        self.model = None        
        self.label_mapping = None
        self.n_classes = None
        self.history = None
        self.metrics = None #[tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score'), 'accuracy']
        log_dir = f"logs/fit/run_only_once" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        decay_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001 ,min_lr=0.00001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
        self.callbacks = [tensorboard_callback, decay_rate, early_stopping]


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
        
    def insert_values(self,train_path,test_path):    
        def insert(df):
            
            # initialize x self.and y matrices
            num_lines = len(df)
            self.n_classes = df['class'].nunique()          
            x_matrix = np.zeros((num_lines, self.max_seq_len ,300))
            y_matrix = np.zeros((num_lines, self.n_classes))


            # insert values
            for i, row in df.iterrows():
                label = row[0]
                sentence = row[1]
                if isinstance(sentence, str):
                    words = sentence.split()[:self.max_seq_len]
                    for j, word in enumerate(words):
                        if word in self.w2v:
                            x_matrix[i, j, :] = self.w2v[word]
                else:
                    continue        
                y_matrix[i,label] = 1.0    

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
        self.build_cnn()  
        self.history = self.model.fit(train_x, train_y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(val_x, val_y), callbacks=self.callbacks, verbose=0)
        return self.history
    def evaluate(self,test_x, test_y):
        return self.model.evaluate(test_x, test_y,return_dict=True)



    def run_n_times(self,train_x, train_y, test_x, test_y, val_x, val_y, dataset_name, n=3):
            hist_dict = {}
            res_dict = {}
            best_val_loss = float('inf')
            for i in range(n):
                print(f'Run {i+1} of {n}')
                self.fit(train_x, train_y, val_x, val_y)
                res = self.evaluate(test_x, test_y)
                res_dict[i+1] = res
                if self.history.history['val_loss'][-1] < best_val_loss:
                    best_val_loss = self.history.history['val_loss'][-1]
                    self.model.save(f"models/{dataset_name}_best_model.h5")
                self.model.set_weights([np.zeros(w.shape) for w in self.model.get_weights()])
            
            avg_dict = {metric: round(sum(values[metric] for values in res_dict.values()) / len(res_dict), 4)  for metric in res_dict[1].keys()}
            
            # Save the average results to disk
            os.makedirs("results", exist_ok=True)
            with open(f"results/{dataset_name}_avg_results.txt", "w") as f:
                for key, value in avg_dict.items():
                    f.write(f"{key}: {value}\n")
            
            K.clear_session()
            
            return hist_dict, res_dict, avg_dict

    

# if __name__ == '__main__':
#     train_path  = 'data/original/trec/train.csv'
#     test_path   = 'data/original/trec/test.csv'
#     w2v_path = 'w2v.pkl'
#     max_seq_len = 150
#     batch_size = 8
#     epochs = 30
#     model = CNN(dims=300, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs, w2v_path=w2v_path)
#     train_x, train_y, test_x, test_y, val_x, val_y, n_classes = model.insert_values(train_path,test_path)
#     his,res,avg = model.run_n_times(train_x, train_y, test_x, test_y, val_x, val_y, n=3)
#     print (avg)