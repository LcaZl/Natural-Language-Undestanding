import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import *

class SBJ_Model(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, vocab_size, num_layers=1, dropout=0, bidirectional=False):
        super(SBJ_Model, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_TOKEN)

        # Define the LSTM layer
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        # Define a dropout layer
        self.dropout = nn.Dropout(dropout)

        # Define the fully connected layer
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * factor, output_size)
        
    def forward(self, x, lengths):
        # x shape: (batch_size, seq_length)
        # lengths shape: (batch_size)

        # Embed the input
        emb = self.embedding(x)

        # Pack the batch
        packed_input = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # Process with LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Unpack the batch
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply dropout
        output = self.dropout(output)

        # We take the output from the final time-step of each sequence
        output = output[torch.arange(output.size(0)), lengths - 1]
        
        # Pass the output through the fully connected layer
        output = self.fc(output)
        
        return output
