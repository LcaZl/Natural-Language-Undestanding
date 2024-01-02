from functions import *

class LM_Model(nn.Module):
    def __init__(self, output_size, dropout = 0.1):#, vader=False):
        
        super(LM_Model, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):#, vader_scores):
        # input_ids => [batch size, max_seq_length]
        # attention_mask => [batch size, max_seq_length]
        # print(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # Last bert layer

        hidden = self.dropout(hidden)
        output = self.fc(hidden[:, 0, :])  # output of token [CLS]
        return output
