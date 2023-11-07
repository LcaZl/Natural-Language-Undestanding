import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import *

class SUBJ_Model(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, vocab_size, num_layers=1, dropout=0, bidirectional=False):
        super(SUBJ_Model, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_TOKEN)

        # Sostituire LSTM con RNN
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        
        # Definire il fully connected layer
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        
        # Definire un dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text => [batch size, sent len]
        # text_lengths => [batch size]

        embedded = self.dropout(self.embedding(text))  # embedded => [batch size, sent len, emb dim]

        # Pack sequence
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)  # hidden non contiene gli stati delle celle

        # Unpack sequence
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # Applichiamo il dropout
        output = self.dropout(output)

        # Utilizziamo la concatenazione degli stati nascosti forward e backward per RNN bidirezionali
        # o semplicemente l'ultimo stato nascosto per RNN unidirezionali
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]

        # hidden => [batch size, hid dim]
        return self.fc(hidden)  # [batch size, output dim]
