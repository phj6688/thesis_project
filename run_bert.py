from simple_bert import SimpleBert
import numpy as np

np.random.seed(100)

dataset_list = ['cr', 'trec', 'agnews', 'pc', 'yelp', 'kaggle_med', 'cardio', 'bbc', 'sst2','subj']
#dataset_list = ['agnews']


if __name__ == "__main__":   
    
    
    #text = 'not too many great features'
    for dataset in dataset_list:
        print(f'Running {dataset} dataset')
        simple_bert = SimpleBert(dataset)
        print(f"Loaded data for {dataset} dataset")
        simple_bert.load_data()
        print(f"Trained model for {dataset} dataset")
        simple_bert.train_model()
        print(f"Evaluated model for {dataset} dataset")
        res = simple_bert.evaluate_model()                
        # model will be saved during the training process
        simple_bert.save_results(f"results/bert/{dataset}_result.txt")
        print(f"Saved model and results for {dataset} dataset")
        print('cleaning up the checkpoint folders')
        simple_bert.clean_up()
        print(f'results: \n\n\n{res}\n\n\n')


        #pre_last_layer_output = simple_bert.extract_pre_last_layer(text)
        #print(pre_last_layer_output)