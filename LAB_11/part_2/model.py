import torch
import torch.nn as nn
import torch.optim as optim

# Definizione della rete neurale
class AspectTermExtractor(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(AspectTermExtractor, self).__init__()
        self.hidden_dim = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, text, lengths):
        # Embedding layer
        embedded = self.embedding(text)

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # LSTM layer
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Fully connected layer
        logits = self.fc(output)

        # Output
        return logits