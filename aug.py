from functions import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(filename='aug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


fraction = 10
pct_words_to_swap = 0.5
example = 1

dict_datasets = {
    'cr': f'./data/original/cr/train_{fraction}.csv'
    'kaggle_med': f'./data/original/kaggle_med/train_{fraction}.csv',
    'trec': f'./data/original/trec/train_{fraction}.csv',
    'sst2': f'./data/original/sst2/train_{fraction}.csv',
    'subj': f'./data/original/subj/train_{fraction}.csv',
    'yelp': f'./data/original/yelp/train_{fraction}.csv',
    'bbc': f'./data/original/bbc/train_{fraction}.csv',
    'pc': f'./data/original/pc/train_{fraction}.csv',
    'agnews': f'./data/original/agnews/train_{fraction}.csv',
    'cardio': f'./data/original/cardio/train_{fraction}.csv'
}


dict_methods = {
    'aeda': 'aeda_augmenter',
    'backtranslation': 'backtranslation_augmenter'
     'checklist': 'checklist_augmenter',
     'clare': 'clare_augmenter',
     'eda': 'eda_augmenter',
     'wordnet': 'wordnet_augmenter',
     'charswap': 'charswap_augmenter',
     'deletion': 'deletion_augmenter',
     'embedding': 'embedding_augmenter'
 }



for j in tqdm(dict_datasets):
    for i in tqdm(dict_methods):
        print('------------------------------------')
        print(f'Augmenting {j} with {i} method')
        print('------------------------------------')

        logging.debug(f'Augmenting {j} with {i} method')

        df = load_data(dict_datasets[j])
        method = dict_methods[i]
        try:
            result = augment_text(df, method, fraction=1, pct_words_to_swap=pct_words_to_swap,
                                        transformations_per_example=example, label_column='class', target_column='text',
                                        include_original=True)
        
            result = result[['class', 'text']]
            result.to_csv(f'./data/augmented/{j}/frac_{fraction}_meth_{i}_pctwts_{pct_words_to_swap}_example_{example}.csv', index=False)
            print(f'Augmenting done')
            logging.debug(f'Augmenting done')
        except:
            print(f'Error in augmenting {j} with {i} method')
            logging.debug(f'Error in augmenting {j} with {i} method')
            pass
#datasetName_percentage_augmentationMethod_augmentationFactor
