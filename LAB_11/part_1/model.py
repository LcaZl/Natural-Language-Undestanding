from transformers import BertModel
import torch.nn as nn
import torch

class SUBJ_Model(nn.Module):
    def __init__(self, hidden_size, output_size, dropout = 0.1, vader=False):
        super(SUBJ_Model, self).__init__()

        self.vader = vader
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Definire il fully connected layer
        fc_input_size = 768  # Dimensione dell'output di BERT base uncased
        if vader:
            fc_input_size += hidden_size  # Aggiunta per il vader score
        self.fc = nn.Linear(fc_input_size, output_size)
        # Definire un dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, vader_scores):
        # input_ids => [batch size, max_seq_length]
        # attention_mask => [batch size, max_seq_length]
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # Estrai l'ultimo stato nascosto da BERT

        # Applica dropout
        hidden = self.dropout(hidden)

        # Utilizza la concatenazione dell'ultimo stato nascosto con le feature vader (se necessario)
        if self.vader:
            vader_scores = vader_scores.float()
            hidden = torch.cat((hidden, vader_scores), dim=2)

        # Calcola l'output finale utilizzando il fully connected layer
        return self.fc(hidden[:, 0, :])  # Utilizza solo l'output corrispondente al token [CLS]
