from cnn3 import *


if __name__ == '__main__':
    train_path  = 'data/original/trec/train.csv'
    test_path   = 'data/original/trec/test.csv'
    w2v_path = 'w2v.pkl'
    dataset_name = 'trec'
    max_seq_len = 150
    batch_size = 8
    epochs = 30
    model = CNN(dims=300, max_seq_len=max_seq_len, batch_size=batch_size, epochs=epochs, w2v_path=w2v_path)
    train_x, train_y, test_x, test_y, val_x, val_y, n_classes = model.insert_values(train_path,test_path)
    his,res,avg = model.run_n_times(train_x, train_y, test_x, test_y, val_x, val_y,dataset_name, n=3)
    print (avg)