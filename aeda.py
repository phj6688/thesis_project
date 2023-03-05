import random
from collections import namedtuple

class Aeda_Augmenter:
	'''
	AEDA (An Easier Data Augmentation Technique) is a text data augmentation 
	technique for text classification. It was proposed by Karimi, Rossi, and Prati from
	 the IMP Lab at the University of Parma, Italy. The method inserts punctuation marks 
	 into the text of the input sentence. It generates new sentences by selecting some words 
	 in the original sentence and inserting punctuation marks before them. The frequency of inserted
	  punctuation marks is determined by a probability parameter. The augmentation process is repeated
	   several times to generate multiple new sentences. https://arxiv.org/abs/2108.13230
	'''

	def __init__(self,pct_words_to_swap=0.1, transformations_per_example=4):

		self.transformations_per_example = transformations_per_example
		self.pct_words_to_swap = pct_words_to_swap
        

		

	def insert_punctuation_marks(self,sentence):
		PUNCTUATIONS = ('.', ',', '!', '?', ';', ':')
		words = str(sentence).split(' ')
		new_line = []
		q = random.randint(1, int(self.pct_words_to_swap * len(words) + 1))
		qs = random.sample(range(0, len(words)), q)

		for j, word in enumerate(words):
			if j in qs:
				new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
				new_line.append(word)
			else:
				new_line.append(word)
		new_line = ' '.join(new_line)
		return new_line

	def augment(self,line):
		res = []
		for i in range(self.transformations_per_example):		
			new_line = self.insert_punctuation_marks(line)
			res.append(new_line)
		return res


