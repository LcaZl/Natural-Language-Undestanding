# Add functions or classes used for data loading and preprocessing
# From LAB_09
from functions import *

def load_dataset():
    """
    Load dataset and create the lang, vocabulary and loaders for the models.
    """

    # Folder to save weights of the models
    models_weight_folder = 'models_weight/'
    if not os.path.exists(models_weight_folder):
        os.makedirs(models_weight_folder)

    train_raw = read_file("../dataset/ptb.train.txt")
    dev_raw = read_file("../dataset/ptb.valid.txt")
    test_raw = read_file("../dataset/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>","<UNK>"])
    vocab_len = len(lang.word2id)

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    print('\nDataset info:')
    print(f' - Training size: {len(train_raw)}')
    print(f' - Dev size: {len(dev_raw)}')
    print(f' - Test size: {len(test_raw)}')

    return train_dataset, dev_dataset, test_dataset, vocab_len, lang

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line + eos_token)
    return output

def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0 
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}  # inverse dictionary
        
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        # The source and target lists will store the input and target sequences, respectively.
        self.source = []
        self.target = []
        
        # Iterate through each sentence in the corpus
        for sentence in corpus:
            # Add to the source list all tokens except the last one
            self.source.append(sentence.split()[0:-1]) 
            # Add to the target list all tokens except the first one
            self.target.append(sentence.split()[1:]) 
        
        # Convert the source and target sequences to numerical format
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        # Return the total number of examples in the dataset
        return len(self.source)

    def __getitem__(self, idx):
        # Convert source and target sequences to PyTorch tensors and return them
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    # Auxiliary methods
    
    def mapping_seq(self, data, lang): 
        # Convert sequences of tokens to sequences of corresponding numerical identifiers.
        res = []
        unk_id = lang.word2id['<UNK>']

        for seq in data:
            tmp_seq = []
            for x in seq:
                # Check if token is in the vocabulary
                if x in lang.word2id:
                    # Add the token's numerical identifier to tmp_seq
                    tmp_seq.append(lang.word2id[x])
                else:
                    tmp_seq.append(unk_id)
            # Add the converted sequence to res
            res.append(tmp_seq)
        return res

def collate_fn(data, pad_token):

    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item