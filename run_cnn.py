from  cnn import CNN

dataset_list = ['cardio']
#dataset_list = ['bbc']

#dataset_list = ['yelp','bbc','sst2','subj','agnews','trec','pc','cr','kaggle_med','cardio']
#dataset_list = ['pc','cr','kaggle_med','cardio']

if __name__ == '__main__':
    for name in dataset_list:
        try:
            print(f'Running {name} dataset')
            train_path = f'data/original/{name}/train.csv'
            test_path = f'data/original/{name}/test.csv'
            w2v_path = 'w2v.pkl'
            dataset_name = f'{name}'
            max_seq_len = 128
            batch_size = 4
            epochs = 20
            cnn = CNN(dims=300, w2v_path=w2v_path, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs, chunk_size=1000)
            train_dataset, test_dataset, val_dataset, n_classes = cnn.insert_values(train_path, test_path)  # Updated to return datasets
            hist_dict, res_dict, avg_dict = cnn.run_n_times(train_dataset, test_dataset, val_dataset, name, n=3)  # Updated to use datasets
            print('---------------------------------------------------')
            print(f'Average results for {name} dataset')
            print(avg_dict)
            print('---------------------------------------------------')
        except:
            print(f'Error in {name}')
            continue
