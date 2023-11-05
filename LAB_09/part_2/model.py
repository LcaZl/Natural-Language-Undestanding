import torch.nn as nn
import torch

class LM_(nn.Module):
    def __init__(self, nn_type, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dp_prob  = 0, # Output dropout layer probabilities
                 emb_dp_prob = 0, # Embeddings dropout layer probabilities
                 n_layers=1, 
                 weight_tying = False, # Weight tying regularization
                 variational_dropout = False,
                 max_len = 100): # Variation dropout
        
        super(LM_, self).__init__()
        
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        if nn_type == 'RNN': # Pytorch's RNN layer
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False)

        elif nn_type == 'LSTM': # Pytorch's LSTM layer
            self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)

        self.embedding_dropout = nn.Dropout(emb_dp_prob)
        self.output_dropout = nn.Dropout(out_dp_prob)
        
        # Save padding token index
        self.pad_token = pad_index
        
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

        if weight_tying: # If weight tying is enabled share the same weights bewtween embeddings and output
            self.output.weight = self.embedding.weight

        # Initialize variables for variational dropout
        self.variational_dropout = variational_dropout
        self.vd_mask = None
        self.max_seq_length = max_len

    def forward(self, input_sequence):
        # Transform token ids into indexes
        emb = self.embedding(input_sequence)

        if self.variational_dropout:
            mask = self.generate_vd_mask(emb, emb.size(0))
            emb = emb * mask / (1 - self.embedding_dropout.p)
            
        else:
            emb = self.embedding_dropout(emb)            

        rnn_out, _ = self.rnn(emb)

        if self.variational_dropout:
            rnn_out = rnn_out * mask[:, :, :rnn_out.size(2)] / (1 - self.output_dropout.p)

        else:
            rnn_out = self.output_dropout(rnn_out)

        output = self.output(rnn_out).permute(0, 2, 1)
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
    
    def generate_vd_mask(self, emb, batch_size):
        _, current_seq_length, _ = emb.shape
        
        if self.vd_mask is None or self.vd_mask.shape[1] < current_seq_length or self.vd_mask.shape[0] != batch_size:
            # Adjust mask generation to use batch_size and ensure it has the correct shape
            self.vd_mask = torch.bernoulli(torch.full((batch_size, self.max_seq_length, emb.size(2)), 1-self.embedding_dropout.p)).to(emb.device) 
        
        return self.vd_mask[:, :current_seq_length, :]

    
    def reset_vd_mask(self):
        self.vd_mask = None
