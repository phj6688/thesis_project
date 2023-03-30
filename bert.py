import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from datetime import datetime
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

class BERT:
    def __init__(self, model_name, max_seq_len=128, batch_size=16, epochs=5):
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.n_classes = None
        self.history = None
        self.callbacks = None

    def build_bert(self):
        self.model = TFBertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.n_classes)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        #metrics = ['accuracy', tf.keras.metrics.AUC(name='auc'), tfa.metrics.F1Score(self.n_classes, average='weighted', name='f1_score')]
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def prepare_dataset(self, df):
        def generator():
            for _, row in df.iterrows():
                label = row[0]
                sentence = row[1]
                inputs = self.tokenizer(sentence, max_length=self.max_seq_len, padding='max_length', truncation=True, return_tensors='tf')
                y = np.zeros(self.n_classes)
                y[label] = 1.0
                yield (inputs['input_ids'][0], inputs['attention_mask'][0]), y


        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (tf.TensorSpec(shape=(self.max_seq_len,), dtype=tf.int32),
                 tf.TensorSpec(shape=(self.max_seq_len,), dtype=tf.int32)),
                tf.TensorSpec(shape=(self.n_classes,), dtype=tf.float32)
            )
        )
        return dataset

    def prepare_data(self, train_path, test_path):
        train_df = pd.read_csv(train_path, header=None, names=['Label', 'Text'], skiprows=1)
        test_df = pd.read_csv(test_path, header=None, names=['Label', 'Text'], skiprows=1)

        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['Label'])

        self.n_classes = len(train_df['Label'].value_counts())

        train_dataset = self.prepare_dataset(train_df).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = self.prepare_dataset(test_df).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = self.prepare_dataset(val_df).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset, test_dataset, val_dataset


    def fit(self, train_dataset, val_dataset):
        self.build_bert()
        self.history = self.model.fit(train_dataset, epochs=self.epochs, validation_data=val_dataset, callbacks=self.callbacks, verbose=1)
        return self.history

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset, return_dict=True)

    def run_n_times(self, train_dataset, test_dataset, val_dataset, dataset_name, n=3):
        log_dir = f"logs/fit/{dataset_name}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        decay_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
        self.callbacks = [tensorboard_callback, decay_rate, early_stopping]

        res_dict = {}
        best_val_loss = float('inf')
        for i in range(n):
            print(f'Run {i+1} of {n}')
            self.fit(train_dataset, val_dataset)
            res = self.evaluate(test_dataset)
            res_dict[i+1] = res
            if self.history.history['val_loss'][-1] < best_val_loss:
                best_val_loss = self.history.history['val_loss'][-1]
                self.model.save_pretrained(f"models/bert/{dataset_name}_best_model")

        avg_dict = {metric: round(sum(values[metric] for values in res_dict.values()) / len(res_dict), 4) for metric in res_dict[1].keys()}

        # Save the average results to disk
        os.makedirs("results/bert", exist_ok=True)
        with open(f"results/bert/{dataset_name}_avg_results.txt", "w") as f:
            for key, value in avg_dict.items():
                f.write(f"{key}: {value}\n")

        tf.keras.backend.clear_session()

        return res_dict, avg_dict

