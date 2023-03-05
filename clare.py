import torch
import transformers


class Clare_Augmenter:
    def __init__(self, pct_words_to_swap=0.1, transformations_per_example=1):
        self.pct_words_to_swap = pct_words_to_swap
        self.transformations_per_example = transformations_per_example
        



    def augment(self,text):
        # Load pre-trained BERT model and tokenizer
        model_name = 'bert-base-uncased'
        tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
        model = transformers.BertForMaskedLM.from_pretrained(model_name)
        model.eval()

        # Preprocess text
        text = text.replace('#', '')  # Remove special characters
        text = text.strip()  # Remove leading/trailing whitespace

        # Tokenize input text
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

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
                    # Replace token with predicted token from BERT model
                    with torch.no_grad():
                        predictions = model(torch.tensor([token_ids])).logits[0]
                        probabilities = predictions[i].softmax(dim=0)
                        predicted_token_id = torch.multinomial(probabilities, 1).item()
                        predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
                        new_tokens.append(predicted_token)
                else:
                    new_tokens.append(tokenizer.convert_ids_to_tokens([token_id])[0])

            new_text = tokenizer.convert_tokens_to_string(new_tokens)
            new_texts.append(new_text)

        return new_texts
