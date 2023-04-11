import pandas as pd
import numpy as np
import os
import shutil
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from sklearn.model_selection import train_test_split
#from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import torch
from torch.utils.data import Dataset

#set seed to 100 
np.random.seed(100)



# disable wandb
os.environ["WANDB_DISABLED"] = "true"

class Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class BERT:
    def __init__(self,train_path,test_path,trainings_arguments:TrainingArguments, model_name='distilbert-base-uncased'):
#        self.max_seq_len = max_seq_len
        self.model_name = model_name
        self.n_runs = None
        self.best_val_loss = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.trainings_arguments = trainings_arguments        
        self.train_dataset, self.test_dataset, self.n_classes = self.prepare_dataset(train_path, test_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.n_classes)
        self.compute_metrics_func = self.compute_metrics
        self.trainer = Trainer(
            model=self.model,
            args=self.trainings_arguments,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics_func)    
               
    def prepare_dataset(self, train_path, test_path):
        train_data = pd.read_csv(train_path).sample(frac=1).reset_index(drop=True)
        test_data = pd.read_csv(test_path).sample(frac=1).reset_index(drop=True)
        n_classes = len(train_data['class'].unique())        
        # encode the labels
        encoder = LabelEncoder()
        train_data['class'] = encoder.fit_transform(train_data['class'])
        test_data['class'] = encoder.transform(test_data['class'])
        # Remove rows with missing or invalid 'text' values
        train_data = train_data[train_data['text'].apply(lambda x: isinstance(x, str))]
        test_data = test_data[test_data['text'].apply(lambda x: isinstance(x, str))]        
        
        # tokenize the text
        train_encodings = self.tokenizer(train_data['text'].tolist(), truncation=True, padding=True)#, #max_length=self.max_seq_len)
        test_encodings = self.tokenizer(test_data['text'].tolist(), truncation=True, padding=True)#, #max_length=self.max_seq_len)
        # create dataset
        train_dataset = Dataset(train_encodings, train_data['class'].tolist())
        test_dataset = Dataset(test_encodings, test_data['class'].tolist())
        return train_dataset, test_dataset, n_classes

    def compute_metrics(self,pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        probs = softmax(pred.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        accuracy = accuracy_score(labels, preds)
        
        # Calculate AUC
        if len(np.unique(labels)) > 2:  # Multi-class case
            auc = roc_auc_score(labels, probs, multi_class="ovo", average="weighted")
        else:  # Binary case
            auc = roc_auc_score(labels, probs[:, 1])  # Use the probability of the positive class

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }
    
    def train_and_evaluate(self, run_idx, dataset_name):
        print(f'Run {run_idx} of {self.n_runs}')
        self.trainer.train()
        self.trainer.save_model(f"models/bert/{dataset_name}_run_{run_idx}_model")
        results = self.trainer.evaluate()
        if results['eval_loss'] < self.best_val_loss:
            self.best_val_loss = results['eval_loss']
            self.trainer.save_model(f"models/bert/{dataset_name}_best_model")
        self.model.init_weights()
        return results

    def clean_up_models(self, dataset_name):
        for i in range(self.n_runs):
            shutil.rmtree(f"models/bert/{dataset_name}_run_{i+1}_model")

    def calculate_and_save_averages(self, res_dict, dataset_name):
        avg_dict = {metric: round(sum(values[metric] for values in res_dict.values()) / len(res_dict), 4) for metric in res_dict[1].keys()}
        order = ['loss', 'auc', 'f1', 'accuracy']
        filtered_metrics = {key: avg_dict[i] for i in avg_dict if i.split('_')[-1] in order}
        os.makedirs("results/bert", exist_ok=True)
        with open(f"results/bert/{dataset_name}_avg_results.txt", "w") as f:
            for key, value in filtered_metrics.items():
                f.write(f"{key}: {value}\n")
        return filtered_metrics

    def run_n_times(self, dataset_name, n=3):
        self.n_runs = n
        self.best_val_loss = float('inf')
        res_dict = {}
        
        for i in range(n):
            res_dict[i+1] = self.train_and_evaluate(i+1, dataset_name)
            
        self.clean_up_models(dataset_name)
        avg_metrics = self.calculate_and_save_averages(res_dict, dataset_name)
        new_dict = {}
        order = ['loss', 'auc', 'f1','accuracy']
        for i in avg_metrics:
            if i.split('_')[-1] in order:
                key = i.split('_')[-1]
                new_dict[key] = avg_metrics[i]

        os.makedirs("results/bert", exist_ok=True)        
        with open(f"results/bert/{dataset_name}_avg_results.txt", "w") as f:
            for key, value in new_dict.items():
                f.write(f"{key}: {value}\n")

        return new_dict
        



