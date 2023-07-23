from functions import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('aug.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

pct_words_to_swap = 0.5
example = 4

dict_datasets = {
    # 'cr': './data/original/cr/train.csv',
    # 'kaggle_med': './data/original/kaggle_med/train.csv',
    # 'trec': './data/original/trec/train.csv',
    # 'sst2': './data/original/sst2/train.csv',
    # 'subj': './data/original/subj/train.csv',
    #'yelp': './data/original/yelp/train.csv',
    # 'bbc': './data/original/bbc/train.csv',
     #'pc': './data/original/pc/train.csv',
     'agnews': './data/original/agnews/train.csv',
    # 'cardio': './data/original/cardio/train.csv',
    #'pubmed': './data/original/pubmed/train.csv',

}


dict_methods = {
    # 'aeda': 'aeda_augmenter',
    # 'checklist': 'checklist_augmenter',
    # 'eda': 'eda_augmenter',
    #'wordnet': 'wordnet_augmenter',
    #'charswap': 'charswap_augmenter',
    #'deletion': 'deletion_augmenter',
    'embedding': 'embedding_augmenter',
    'clare': 'clare_augmenter',
    'backtranslation': 'backtranslation_augmenter'
    }

for dataset_name, dataset_path in dict_datasets.items():
    for method_name, method_func_name in dict_methods.items():
        logger.info(f'Starting augmentation for {dataset_name} dataset using {method_name} method')
        try:
            df = load_data(dataset_path)
            logger.info(f'Loaded {dataset_name} dataset with {len(df)} rows successfully')

            # Initialize augmenter only once
            augmenter = get_augmenter(method_func_name, pct_words_to_swap, example)

            result = augment_text(df, augmenter, fraction=1, pct_words_to_swap=pct_words_to_swap,
                                  transformations_per_example=example, label_column='class', target_column='text',
                                  include_original=True)

            num_examples_augmented = len(result)
            result = result[['class', 'text']]
            result.to_csv(f'./data/augmented/{dataset_name}/meth_{method_name}_pctwts_{pct_words_to_swap}_example_{example}.csv', index=False)
            logger.info(f'Augmentation completed successfully for {dataset_name} dataset using {method_name} method. Augmented {num_examples_augmented} examples.')
        except Exception as e:
            logger.exception(f'Error in augmenting {dataset_name} dataset with {method_name} method: {str(e)}')
