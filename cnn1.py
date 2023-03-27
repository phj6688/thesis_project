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
        #metrics = [tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(n_classes, average='weighted', name='f1_score'), 'accuracy']
        cnn_model = Model(inputs=input_layer, outputs=dense_out)
        cnn_model.compile(optimizer='adam', loss=loss, metrics=self.metrics)
        cnn_model.summary()
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
        val_df = val_df.sample(frac=1).reset_index(drop=True)

        train_x, train_y = insert(train_df)
        test_x, test_y = insert(test_df)
        val_x, val_y = insert(val_df)

        return train_x, train_y, test_x, test_y, val_x, val_y, self.n_classes          

    def fit(self,train_x, train_y,  val_x, val_y):
        self.metrics = [tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score'), 'accuracy']
        self.build_cnn()  
        self.history = self.model.fit(train_x, train_y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(val_x, val_y), callbacks=self.callbacks)
        return self.history
    
    def fit_n_times(self, train_x, train_y, val_x, val_y, runs=3):
        
        for run in range(runs):
            print(f'Run {run + 1} of {runs}')
            self.build_cnn()    
            log_dir = f"logs/fit/run_{run + 1}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            decay_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001 ,min_lr=0.00001)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
            callbacks = [tensorboard_callback, decay_rate, early_stopping]
            self.history = self.model.fit(train_x, train_y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(val_x, val_y), callbacks=callbacks)            
            # Reset the model to initial state for next run
            self.model.set_weights([np.zeros(w.shape) for w in self.model.get_weights()])

        return self.history


    def evaluate(self,test_x, test_y):
        return self.model.evaluate(test_x, test_y,return_dict=True)
    
    def evaluate_n_times(self, test_x, test_y, runs=3):
        results = []
        for run in range(runs):
            print(f'Run {run + 1} of {runs}')
            
            # Evaluate the model on the test data and store the result
            result = self.evaluate(test_x, test_y)
            results.append(result)
        
        # Calculate the average of the results
        avg_result = {}
        for key in results[0].keys():
            avg_result[key] = np.mean([result[key] for result in results], axis=0)
        
        print(f'Average result: {avg_result}')
        return avg_result
    

if __name__ == '__main__':
    train_path  = 'data/original/cr/train.csv'
    test_path   = 'data/original/cr/test.csv'
    w2v_path = 'w2v.pkl'
    max_seq_len = 32
    batch_size = 128
    epochs = 30
    model = CNN(dims=300, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs, w2v_path=w2v_path)
    train_x, train_y, test_x, test_y, val_x, val_y, n_classes = model.insert_values(train_path,test_path)
    # model.fit(train_x, train_y, val_x, val_y)
    # model.evaluate(test_x, test_y)
    model.fit_n_times(train_x, train_y, val_x, val_y, runs=3)
    model.evaluate_n_times(test_x, test_y, runs=3)