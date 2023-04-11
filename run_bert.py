from bert2 import BERT
from transformers import TrainingArguments

dataset_list = ['trec','agnews', 'pc', 'yelp', 'cr', 'kaggle_med', 'cardio', 'bbc', 'sst2','subj']
#dataset_list = ['trec']

if __name__ == '__main__':
    for name in dataset_list:
        try:
            print(f'Running {name} dataset')
            train_path  = f'data/original/{name}/train.csv'
            test_path   = f'data/original/{name}/test.csv'
            model_name = 'distilbert-base-uncased'
            training_args = TrainingArguments(
                output_dir='./results/bert',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                evaluation_strategy="epoch",
                # eval_steps=100,
                logging_steps=10,
                # save_steps=0,
                logging_dir='./logs/bert',
                metric_for_best_model="f1",
                #learning_rate=2e-5,
                seed=100
            )



            bert = BERT(train_path, test_path, training_args, model_name=model_name)            
            avg_dict = bert.run_n_times(name, n=3)
            print('---------------------------------------------------')
            print(f'Average results for {name} dataset')
            print(avg_dict)
            print('---------------------------------------------------')
        except Exception as e:
            print(f'Error in {name}')
            print(str(e))
            continue
