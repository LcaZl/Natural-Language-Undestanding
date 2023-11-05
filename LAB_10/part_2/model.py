from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torchcrf import CRF
import torch

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.1):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class jointBERT(nn.Module):
    def __init__(self, out_slot, out_int, dropout_rate=0.1, use_crf=True):
        super(jointBERT, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Define output layers
        self.intent_classifier = IntentClassifier(self.bert.config.hidden_size, out_int, dropout_rate)
        self.slot_classifier = SlotClassifier(self.bert.config.hidden_size, out_slot, dropout_rate)

        self.use_crf = use_crf
        if use_crf:
            # CRF layer for slot filling
            self.crf = CRF(out_slot, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits

    def decode_slots(self, slot_logits, attention_mask):
        return self.crf.decode(slot_logits, attention_mask.bool())


