import torch.nn as nn
import torch
from utils import *

class LangModel(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dp_prob=0, emb_dp_prob=0, variational_dp_prob=0.5, # Variational dropout probability
                 n_layers=1, weight_tying=False, variational_dropout=False, max_len=100):
        super(LangModel, self).__init__()
        
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)

        self.embedding_dropout = nn.Dropout(emb_dp_prob)
        self.output_dropout = nn.Dropout(out_dp_prob)
        
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

        if weight_tying:
            self.output.weight = self.embedding.weight

        self.variational_dropout = variational_dropout
        self.variational_dp_prob = variational_dp_prob
        self.vd_mask = None
        self.max_seq_length = max_len

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        if self.variational_dropout:
            if self.emb_vd_mask is None:
                self.emb_vd_mask = self.generate_vd_mask(emb.size(2), emb.size(0))[:, :emb.size(1), :]
            emb = emb * self.emb_vd_mask
        else:
            emb = self.embedding_dropout(emb)            

        lstm_out, _ = self.lstm(emb)

        if self.variational_dropout:
            if self.lstm_vd_mask is None:
                self.lstm_vd_mask = self.generate_vd_mask(lstm_out.size(2), lstm_out.size(0))[:, :lstm_out.size(1), :]
            lstm_out = lstm_out * self.lstm_vd_mask
        else:
            lstm_out = self.output_dropout(lstm_out)

        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    
     # Get the embedding for a specific token
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()
    
    # get most similar token to the given vector
    def get_most_similar(self, vector, top_k=10):

        embs = self.embedding.weight.detach().cpu().numpy()
        
        # Cosine similarity
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:  # Evitare il token di padding
                scores.append(cosine_similarity(x.reshape(1, -1), vector.reshape(1, -1))[0][0])
        
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)
    
    def reset_vd_mask(self):
        self.emb_vd_mask = None
        self.lstm_vd_mask = None

    def generate_vd_mask(self, size, batch_size):
        scale = 1 / (1 - self.variational_dp_prob)
        mask = torch.bernoulli(torch.full((batch_size, self.max_seq_length, size), self.variational_dp_prob)).to(DEVICE) * scale
        return mask
