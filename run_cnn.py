from cnn3 import *

dataset_list = ['agnews','pc','yelp','cr','kaggle_med','cardio','bbc','sst2']

if __name__ == '__main__':
    for name in dataset_list:
        try:
            print (f'Running {name} dataset')
            train_path  = f'data/original/{name}/train.csv'
            test_path   = f'data/original/{name}/test.csv'
            w2v_path = 'w2v.pkl'
            dataset_name = f'{name}'
            max_seq_len = 150
            batch_size = 4
            epochs = 30
            model = CNN(dims=300, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs, w2v_path=w2v_path)
            train_x, train_y, test_x, test_y, val_x, val_y, n_classes = model.insert_values(train_path,test_path)
            his,res,avg = model.run_n_times(train_x, train_y, test_x, test_y, val_x, val_y,dataset_name, n=3)
            print ('---------------------------------------------------')
            print (f'Average results for {name} dataset')
            print (avg)
            print ('---------------------------------------------------')
        except:
            print (f'Error in {name}')
            continue