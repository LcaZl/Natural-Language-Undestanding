import torch.nn as nn

class LM_(nn.Module):
    def __init__(self, nn_type, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dp_prob  = 0, emb_dp_prob = 0, n_layers=1):

        super(LM_, self).__init__()
        
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        if nn_type == 'RNN': # Pytorch's RNN layer
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False)

        elif nn_type == 'LSTM': # Pytorch's LSTM layer
            self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)

        self.embedding_dropout = nn.Dropout(emb_dp_prob)

        
        # Save padding token index
        self.pad_token = pad_index
        
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        self.output_dropout = nn.Dropout(out_dp_prob)
        
    def forward(self, input_sequence):
        # Transform token ids into indexes
        emb = self.embedding(input_sequence)
        
        emb = self.embedding_dropout(emb)
            
        rnn_out, _ = self.rnn(emb)
        
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
