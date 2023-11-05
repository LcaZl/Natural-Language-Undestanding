import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import *

class SUBJ_Model(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, vocab_size, num_layers=1, dropout=0, bidirectional=False):
        super(SUBJ_Model, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_TOKEN)

        # Define the LSTM layer
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)        
        # Define a dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text => [batch size, sent len]
        # text_lengths => [batch size]

        embedded = self.dropout(self.embedding(text))  # embedded => [batch size, sent len, emb dim]

        # pack sequence
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # We apply the dropout
        output = self.dropout(output)

        # We use the concatenation of the forward and backward hidden states for bidirectional LSTMs
        # or just the last hidden state for unidirectional LSTMs
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        # hidden => [batch size, hid dim]
        return self.fc(hidden.squeeze(0))  # [batch size, output dim]

