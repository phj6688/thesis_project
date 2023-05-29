# import torch

# print("GPU available:", torch.cuda.is_available())
# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)
# print("cuDNN version:", torch.backends.cudnn.version())


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("gpu availability: ", tf.test.is_gpu_available())

print('=====================')
from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())


# import numpy as np
# def load_glove_embeddings(path):
#     word2vec_dict = {}            
#     with open(path, 'r', encoding='utf-8') as file:
#         for line in file:
#             try:
#                 values = line.split()
#                 word = values[0]
#                 vector = np.asarray(values[1:], dtype='float32')
#                 word2vec_dict[word] = vector
#             except ValueError:                
#                 continue    
#     return word2vec_dict

# path = "glove.840B.300d.txt"
# word2vec = load_glove_embeddings(path)
# print(len(word2vec))


# import pickle

# with open('w2v.pkl', 'wb') as f:
#     pickle.dump(word2vec, f)