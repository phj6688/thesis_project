from lstm import *

dataset_list = ['agnews','pc','yelp','cr','kaggle_med','cardio','bbc','sst2']

if __name__ == '__main__':
    for name in dataset_list:
        try:
            print (f'Running {name} dataset')
            train_path  = f'data/original/{name}/train.csv'
            test_path   = f'data/original/{name}/test.csv'
            w2v_path = 'w2v.pkl'
            dataset_name = f'{name}'
            max_seq_len = 64
            batch_size = 8
            epochs = 30
            lstm = LSTM(dims=300, w2v_path=w2v_path, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs)
            train_x, train_y, test_x, test_y, val_x, val_y, n_classes = lstm.insert_values(train_path,test_path)
            hist_dict, res_dict, avg_dict = lstm.run_n_times(train_x, train_y, test_x, test_y, val_x, val_y, name, n=3)
            print ('---------------------------------------------------')
            print (f'Average results for {name} dataset')
            print (avg_dict)
            print ('---------------------------------------------------')
        except:
            print (f'Error in {name}')
            continue


