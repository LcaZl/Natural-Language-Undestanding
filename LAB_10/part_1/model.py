from functions import *

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, bidirectional = False, dropout_prob = 0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (ouput size for intent class)
        # emb_size = word embedding size

        self.bidirectional = bidirectional

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=PAD_TOKEN)

        # LSTM Layer
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional)
        
        # Define output size depending on bidirectional
        hid_size = (2 * hid_size) if bidirectional else hid_size
        
        # Linear Layers for slot and intent output
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

        # Dropout Layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance) 
        utt_emb = utt_emb.permute(1,0,2) 

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), enforce_sorted=False)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        
        # Managing bidirectional
        if self.bidirectional:
            last_hidden = torch.cat((last_hidden[0,:,:], last_hidden[1,:,:]), dim=1)
        else:
            last_hidden = last_hidden[-1,:,:]

        # Dropout if defined
        utt_encoded = self.dropout(utt_encoded)
        last_hidden = self.dropout(last_hidden)

        # Slot and Intent logits computation
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        slots = slots.permute(1,2,0) 

        return slots, intent