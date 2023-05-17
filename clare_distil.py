import torch
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Clare_Augmenter:
    def __init__(self, pct_words_to_swap=0.1, transformations_per_example=1):
        self.pct_words_to_swap = pct_words_to_swap
        self.transformations_per_example = transformations_per_example
        self.model_name = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def augment(self, text):
        # Preprocess text
        text = text.replace('#', '')  # Remove special characters
        text = text.strip()  # Remove leading/trailing whitespace

        # Tokenize input text
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Select indices of words to perturb
        num_words = len(token_ids)
        indices_to_modify = set()
        while len(indices_to_modify) < int(num_words * self.pct_words_to_swap):
            indices_to_modify.add(torch.randint(num_words, size=(1,)).item())

        # Perturb words
        new_texts = []
        for _ in range(self.transformations_per_example):
            new_tokens = []
            for i, token_id in enumerate(token_ids):
                if i in indices_to_modify:
                    # Replace token with predicted token from DistilBert model
                    with torch.no_grad():
                        input_tensor = torch.tensor([token_ids]).to(self.device)
                        predictions = self.model(input_tensor).logits[0]
                        probabilities = predictions[i].softmax(dim=0)
                        predicted_token_id = torch.multinomial(probabilities, 1).item()
                        predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
                        new_tokens.append(predicted_token)
                else:
                    new_tokens.append(self.tokenizer.convert_ids_to_tokens([token_id])[0])

            new_text = self.tokenizer.convert_tokens_to_string(new_tokens)
            new_texts.append(new_text)

        return new_texts
