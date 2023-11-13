from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class jointBERT(nn.Module):
    def __init__(self, output_aspects, output_polarities, dropout_rate=0.1):
        super(jointBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout_rate)
        self.aspect_linear = nn.Linear(self.bert.config.hidden_size, output_aspects)
        self.polarity_linear = nn.Linear(self.bert.config.hidden_size, output_polarities)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        aspect_logits = self.aspect_linear(sequence_output)
        polarity_logits = self.polarity_linear(sequence_output)
        return aspect_logits, polarity_logits



