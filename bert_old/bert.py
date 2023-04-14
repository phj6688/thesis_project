import pandas as pd
import numpy as np
import os
import shutil
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

#set seed to 100 
np.random.seed(100)



# disable wandb
os.environ["WANDB_DISABLED"] = "true"

class Dataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
class BERT:
    def __init__(self,train_path,test_path,trainings_arguments:TrainingArguments, model_name='bert-base-uncased'):
#        self.max_seq_len = max_seq_len
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.trainings_arguments = trainings_arguments        
        self.train_dataset, self.test_dataset, self.n_classes = self.prepare_dataset(train_path, test_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.n_classes)
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
    
    def run_n_times(self, dataset_name, n=3):
        res_dict = {}
        best_val_loss = float('inf')
        for i in range(n):
            print(f'Run {i+1} of {n}')
            self.trainer.train()
            self.trainer.save_model(f"models/bert/{dataset_name}_run_{i+1}_model")
            results = self.trainer.evaluate()
            res_dict[i+1] = results
            if results['eval_loss'] < best_val_loss:
                best_val_loss = results['eval_loss']
                self.trainer.save_model(f"models/bert/{dataset_name}_best_model")
            self.model.init_weights()
            # delete the model folders to save space except for the best model
        for i in range(n):
            shutil.rmtree(f"models/bert/{dataset_name}_run_{i+1}_model")
            

        avg_dict = {metric: round(sum(values[metric] for values in res_dict.values()) / len(res_dict), 4) for metric in res_dict[1].keys()}

        # Save the average results to disk
        # filter the metrics to only include the ones we want        
        new_dict = {}
        order = ['loss', 'auc', 'f1','accuracy']
        for i in avg_dict:
            if i.split('_')[-1] in order:
                key = i.split('_')[-1]
                new_dict[key] = avg_dict[i]

        os.makedirs("results/bert", exist_ok=True)        
        with open(f"results/bert/{dataset_name}_avg_results.txt", "w") as f:
            for key, value in new_dict.items():
                f.write(f"{key}: {value}\n")

        return new_dict


