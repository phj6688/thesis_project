import pandas as pd
import numpy as np
from textattack.augmentation import *
from aeda import Aeda_Augmenter
import os



def load_data(path):
    """
    Loads data from a txt or csv file.
    """
    # Check file format
    if path.endswith('.txt'):
        sep = '|'
    elif path.endswith('.csv'):
        sep = ','
    else:
        raise ValueError('File format not supported.')

    # Load data from file
    df = pd.read_csv(path, sep=sep, header=None, names=['text'])

    # Split text column into class and text columns
    try:
        df['class'] = df['text'].apply(lambda x: x.split('\t')[0])
        df['text'] = df['text'].apply(lambda x: x.split('\t')[1])
    except:
        df['class'] = df['text'].apply(lambda x: x.split(' ', 1)[0])
        df['text'] = df['text'].apply(lambda x: x.split(' ', 1)[-1] if len(x.split(' ', 1)) > 1 else ' ')

    df = df[['class', 'text']]
    return df



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


# def augment_text(df, aug_method, fraction, pct_words_to_swap, transformations_per_example,
#                  label_column='class', target_column='text', include_original=True):
#     '''
#     Augments the text in a Pandas DataFrame using the specified augmentation method.

#     Parameters:
#     df (Pandas DataFrame): The input DataFrame containing the text data.
#     aug_method (str): The name of the augmentation method to use. Must be one of 'eda_augmenter', 
#         'wordnet_augmenter', 'clare_augmenter', 'backtranslation_augmenter', 'aeda_augmenter', 
#         or 'checklist_augmenter'.
#     fraction (float): The fraction of the input data to use for augmentation. Must be between 0 and 1.
#     pct_words_to_swap (float): The percentage of words in each text to replace during augmentation. Must be between 0 and 1.
#     transformations_per_example (int): The number of transformations to apply to each text during augmentation. Must be a positive integer.
#     label_column (str): The name of the label column in the input DataFrame. Defaults to 'class'.
#     target_column (str): The name of the target column in the input DataFrame. Defaults to 'text'.
#     include_original (bool): Whether to include the original text in the output. If True, the output DataFrame will have
#         two rows for each input row, one with the original text and one with the augmented text. If False, the output
#         DataFrame will have only the augmented text. Defaults to True.

#     Returns:
#     Pandas DataFrame: A new DataFrame with the augmented text and labels.

#     Raises:
#     ValueError: If any of the input parameters are invalid.
#     '''
#     # Validate input parameters
#     if not isinstance(df, pd.DataFrame):
#         raise ValueError('df must be a Pandas DataFrame')
#     if not isinstance(aug_method, str):
#         raise ValueError('aug_method must be a string')
#     if aug_method not in ['eda_augmenter', 'wordnet_augmenter', 'clare_augmenter', 'backtranslation_augmenter', 'aeda_augmenter', 'checklist_augmenter']:
#         raise ValueError('aug_method must be one of: eda_augmenter, wordnet_augmenter, clare_augmenter, backtranslation_augmenter, aeda_augmenter, checklist_augmenter')
#     if not 0 < fraction <= 1:
#         raise ValueError('fraction must be between 0 and 1')
#     if not 0 < pct_words_to_swap <= 1:
#         raise ValueError('pct_words_to_swap must be between 0 and 1')
#     if not isinstance(transformations_per_example, int) or transformations_per_example < 1:
#         raise ValueError('transformations_per_example must be a positive integer')
#     if not isinstance(label_column, str):
#         raise ValueError('label_column must be a string')
#     if not isinstance(target_column, str):
#         raise ValueError('target_column must be a string')
#     if not isinstance(include_original, bool):
#         raise ValueError('include_original must be a bool')

#     # Select the appropriate augmenter
#     augmenter_dict = { 
#         'eda_augmenter': EasyDataAugmenter,
#         'wordnet_augmenter': WordNetAugmenter,
#         'clare_augmenter' : CLAREAugmenter,
#         'backtranslation_augmenter': BackTranslationAugmenter,
#         'aeda_augmenter': Aeda_Augmenter,
#         'checklist_augmenter': CheckListAugmenter

#     }

#     augmenter_class = augmenter_dict[aug_method]
#     augmenter = augmenter_class(pct_words_to_swap=pct_words_to_swap, transformations_per_example=transformations_per_example)

#     # Sample a fraction of the input data
#     df = df.sample(frac=fraction)

#     # Augment the text data
#     text_list, class_list = [], []
#     for c, txt in zip(df[label_column], df[target_column]):
#         res = augmenter.augment(txt)
#         if include_original:
#             text_list.append(txt)
#             class_list.append(c)
#             for i in res:
#                 text_list.append(i)
#                 class_list.append(c)
#         else:
#             for i in range(len(res)):
#                 text_list.append(res[i])
#                 class_list.append(c)

#     # Create the output DataFrame
#     df_augmented = pd.DataFrame({target_column: text_list, label_column: class_list})

#     return df_augmented


def augment_text(df,aug_method,fraction,pct_words_to_swap,transformations_per_example,
                label_column='class',target_column='text',include_original=True):

    augmenter_dict = { 
    'eda_augmenter':EasyDataAugmenter(pct_words_to_swap=pct_words_to_swap,
                                    transformations_per_example=transformations_per_example)
                                    ,
    'wordnet_augmenter':WordNetAugmenter(pct_words_to_swap=pct_words_to_swap,
                                    transformations_per_example=transformations_per_example)
                                    ,
    'clare_augmenter' :CLAREAugmenter(pct_words_to_swap=pct_words_to_swap,
                                    transformations_per_example=transformations_per_example)
                                    ,
    'backtranslation_augmenter':BackTranslationAugmenter(pct_words_to_swap=pct_words_to_swap,
                                    transformations_per_example=transformations_per_example)
                                    ,
    'checklist_augmenter' :CheckListAugmenter(pct_words_to_swap=pct_words_to_swap,
                                         transformations_per_example=transformations_per_example)
                                         ,
    # 'embedding_augmenter':EmbeddingAugmenter(pct_words_to_swap=pct_words_to_swap,
    #                                 transformations_per_example=transformations_per_example)
    #                                 ,
    # 'deletion_augmenter':DeletionAugmenter(pct_words_to_swap=pct_words_to_swap,
    #                                 transformations_per_example=transformations_per_example)
    'aeda_augmenter':Aeda_Augmenter(pct_words_to_swap=pct_words_to_swap,
                            transformations_per_example=transformations_per_example)    
    }

    augmenter = augmenter_dict[aug_method]
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