import pandas as pd
import numpy as np
from textattack.augmentation import *
from aeda import Aeda_Augmenter
#from clare import Clare_Augmenter
from clare_distil import Clare_Augmenter
from backtranslation import BackTranslation_Augmenter
import os



def load_data(path):
    """
    Loads data from a txt file.
    """
    # check file format
    if path.endswith('.txt'):
        try:
            df = pd.read_csv(path, sep='|', header=None, names=['text'])
        except:
            df = pd.read_csv(path, sep='\t', header=None, names=['text'])
        try:
            df['class'] = df['text'].apply(lambda x: x.split('\t')[0])
            df['text'] = df['text'].apply(lambda x: x.split('\t')[1])
        except:
            df['class'] = df['text'].apply(lambda x: x.split(' ',1)[0])
            df['text'] = df['text'].apply(lambda x: x.split(' ',1)[1])

        df = df[['class', 'text']]
        return df
    else:
        #raise ValueError('File format not supported.')
        df = pd.read_csv(path,  header=None, names=['class','text'],dtype={'class': int , 'text': str},skiprows=1)
        return df

def csv_to_txt(file):
    '''retun a txt file from a csv file with 
    the same name with class and text separated by a tab'''
    df = pd.read_csv(file)
    df = df[['class','text']]
    np.savetxt(file[:-4] + '.txt', df.values, fmt='%s')

    
def get_augmenter(aug_method, pct_words_to_swap, transformations_per_example):
    augmenter_dict = {
        'eda_augmenter': EasyDataAugmenter,
        'wordnet_augmenter': WordNetAugmenter,
        'clare_augmenter': Clare_Augmenter,
        'backtranslation_augmenter': BackTranslation_Augmenter,
        'checklist_augmenter': CheckListAugmenter,
        'embedding_augmenter': EmbeddingAugmenter,
        'deletion_augmenter': DeletionAugmenter,
        'aeda_augmenter': Aeda_Augmenter,
        'charswap_augmenter': CharSwapAugmenter
    }

    return augmenter_dict[aug_method](pct_words_to_swap=pct_words_to_swap, transformations_per_example=transformations_per_example)

def augment_text(df, augmenter, fraction, pct_words_to_swap, transformations_per_example,
                label_column='class', target_column='text', include_original=True):

    os.system('clear')
    df = df.sample(frac=fraction)
    text_list , class_list = [], []
    for c, txt in zip(df[label_column], df[target_column]):

        res = augmenter.augment(txt)
        if include_original:
            text_list.append(txt)
            class_list.append(c)
            for i in res:
                text_list.append(i)
                class_list.append(c)
        else:
            for i in range(len(res)):
                text_list.append(res[i])
                class_list.append(c)

    df_augmented = pd.DataFrame({target_column: text_list, label_column: class_list})

    return df_augmented


def csv_to_txt(csv_file):
    '''
    Return a txt file from a csv file with the same name
    with class and text separated by a tab.
    '''
    if csv_file.endswith('.csv'):
        df = pd.read_csv(csv_file)
        df = df[['class', 'text']]
        np.savetxt(csv_file[:-4] + '.txt', df.values, fmt='%s', delimiter='\t')
    else:
        raise ValueError('File format not supported.')

def txt_to_csv(txt_file):
    '''
    Return a csv file from a txt file with the same name
    with class and text separated by a comma.
    '''
    if txt_file.endswith('.txt'):
        try:
            df = pd.read_csv(txt_file, sep='\t', header=None, names=['class', 'text'])
        except pd.errors.ParserError:
            df = pd.read_csv(txt_file, sep=' ', header=None, names=['class', 'text'], engine='python')
        df.to_csv(txt_file[:-4] + '.csv', index=False)
    else:
        raise ValueError('File format not supported.')