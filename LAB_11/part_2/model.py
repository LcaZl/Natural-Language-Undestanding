from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

import torch.nn as nn
from transformers import BertModel

class AspectSentimentModel(nn.Module):
    def __init__(self, num_aspect_labels, num_polarity_labels, dropout_rate=0.1):
        super(AspectSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.aspect_classifier = nn.Linear(self.bert.config.hidden_size, num_aspect_labels)
        self.polarity_classifier = nn.Linear(self.bert.config.hidden_size, num_polarity_labels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)

        aspect_logits = self.aspect_classifier(sequence_output)
        
        polarity_logits = self.polarity_classifier(sequence_output)
        return aspect_logits, polarity_logits




