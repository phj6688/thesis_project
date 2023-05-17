from textattack.transformations.sentence_transformations import BackTranslation
from textattack.augmentation import Augmenter


class BackTranslation_Augmenter(Augmenter):
    def __init__(self, pct_words_to_swap=0.1, transformations_per_example=1):
        self.pct_words_to_swap = pct_words_to_swap
        self.transformations_per_example = transformations_per_example
        
    def augment(self, text):
        transformation = BackTranslation()
        augmenter = Augmenter(transformation=transformation, 
                                                pct_words_to_swap=self.pct_words_to_swap, 
                                                transformations_per_example=self.transformations_per_example)
        return augmenter.augment(text)


