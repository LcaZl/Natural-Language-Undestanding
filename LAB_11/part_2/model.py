from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class AspectSentimentModel(nn.Module):
    def __init__(self, num_aspect_labels, num_polarity_labels):
        super(AspectSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Layer di classificazione per gli aspetti
        self.aspect_classifier = nn.Linear(self.bert.config.hidden_size, num_aspect_labels)
        # Layer di classificazione per la polarità
        self.polarity_classifier = nn.Linear(self.bert.config.hidden_size, num_polarity_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Passa gli input attraverso BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        # Applica i layer di classificazione
        aspect_logits = self.aspect_classifier(sequence_output)
        polarity_logits = self.polarity_classifier(sequence_output)
        return aspect_logits, polarity_logits



