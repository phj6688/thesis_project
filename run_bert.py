from bert import BERT

#dataset_list = ['trec','agnews', 'pc', 'yelp', 'cr', 'kaggle_med', 'cardio', 'bbc', 'sst2','subj']
dataset_list = ['trec']

if __name__ == '__main__':
    for name in dataset_list:
        try:
            print(f'Running {name} dataset')
            train_path  = f'data/original/{name}/train.csv'
            test_path   = f'data/original/{name}/test.csv'
            model_name = 'bert-base-uncased'
            max_seq_len = 128
            batch_size = 16
            epochs = 10
            bert = BERT(model_name=model_name, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs)
            train_dataset, test_dataset, val_dataset = bert.prepare_data(train_path, test_path)
            hist_dict, res_dict, avg_dict = bert.run_n_times(train_dataset, test_dataset, val_dataset, name, n=3)
            print('---------------------------------------------------')
            print(f'Average results for {name} dataset')
            print(avg_dict)
            print('---------------------------------------------------')
        except Exception as e:
            print(f'Error in {name}')
            print(str(e))
            continue
