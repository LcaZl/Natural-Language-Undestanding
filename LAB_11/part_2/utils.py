# Add functions or classes used for data loading and preprocessing
import string
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.util import mark_negation
from sklearn.model_selection import KFold
from nltk.sentiment import SentimentIntensityAnalyzer
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
nltk.download('vader_lexicon')
nltk.download('sentiwordnet')
import torch
sia = SentimentIntensityAnalyzer()
# Parameters
PAD_TOKEN = 0
UNK_TOKEN = 1
DEVICE = 'cpu'
TRAIN_PATH = 'dataset/laptop14_train.txt'
TEST_PATH = 'dataset/laptop14_test.txt'
INFO_ENABLED = False

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = file.read()
    dataset = dataset.split('\n')
    dataset = [el.strip() for el in dataset]
    return dataset

def process_raw_data(dataset):
    new_dataset = []
    for sample in dataset:
        if sample:
            raw_sent, words_tagged = sample.split('####')
            
            words_tagged = words_tagged.split()
            tags = [w.split('=')[1] for w in words_tagged]
            words = [w.split('=')[0] for w in words_tagged]
            score = sia.polarity_scores(' '.join(words))['compound']
            if score <= -0.6:
                score = 'VNEG'  # Molto negativo
            elif score <= -0.2:
                score = 'NEG'  # Negativo
            elif score <= 0.2:
                score = 'NET'  # Neutrale
            elif score <= 0.6:
                score = 'POS'  # Positivo
            else:
                score = 'VPOS'  # Molto positivo
            print('vader score for:', ' '.join(words), ' -> ',score)
            assert len(words) == len(tags)

            new_dataset.append((words, score, tags))

    return new_dataset

def load_dataset():
    print(f'\nLoading Dataset Laptop 14...')

    train_raw = read_file(TRAIN_PATH)
    test_raw = read_file(TEST_PATH)

    train_set = process_raw_data(train_raw)
    test_set = process_raw_data(test_raw)

    sents = [el[0] for el in train_set]
    labels = [el[2] for el in train_set]
    vader_labels = [el[1] for el in train_set]

    lang = Lang(sents, labels, vader_labels)

    skf = KFold(n_splits=10, random_state=42, shuffle = True)

    fold_datasets = []  # This will store the dataset splits

    for k, (train_indices, val_indices) in enumerate(skf.split(train_set)):

        train_samples = [train_set[idx] for idx in train_indices]
        val_samples = [train_set[idx] for idx in val_indices]

        train_dataset = Dataset(train_samples, lang)
        val_dataset = Dataset(val_samples, lang)

        train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, collate_fn = collate_fn)
        val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = True, collate_fn = collate_fn)
        
        fold_datasets.append((train_loader, val_loader))
        print(f' - FOLD {k} - Train Size: {len(train_samples)} - Val Size: {len(val_samples)}')

    test_dataset = Dataset(test_set, lang)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = True, collate_fn = collate_fn)


    print(' - Raw sent:', train_raw[0])
    print(' - Training sents:', len(train_raw))
    print(f' - Training len:', len(train_set))
    print(' - Training dataset:', len(train_dataset))
    print(' - Test sents:', len(test_raw))
    print(f' - Test len:', len(test_set))
    print(' - Test dataset:', len(test_dataset))
    print(f' - Training len:', len(train_set))
    print(f' - Labels', lang.label2id)
    print(f' - Train sample:', train_set[0])
    print(f' - Train dataset sample:', train_dataset[0])

    print(f' - Test sample:', test_set[0])
    print(f' - Test dataset sample:', test_dataset[0])
    print('Dataset loaded.\n\n')
    return fold_datasets, test_loader, lang
    
class Lang:
    def __init__(self, sents, labels, vader_labels):
        
        self.word2id = self.mapping_seq(sents, special_token = True)
        self.id2word = {id: word for word, id in self.word2id.items()}

        self.label2id = self.mapping_seq(labels, special_token = False)
        self.id2label = {id: label for label, id in self.label2id.items()}

        self.vlabel2id = self.mapping_seq(vader_labels, special_token = False)
        self.id2vlabel = {id: vlabel for vlabel, id in self.vlabel2id.items()}

        self.vocab_size = len(self.word2id)
        self.label_size = len(self.label2id)

    def encode_sent(self, sentence):
        return [self.word2id.get(word, UNK_TOKEN) for word in sentence]

    def decode_sent(self, sentence_ids):
        return [self.id2word[id] for id in sentence_ids]
    
    def encode_labels(self, sentence):
        return [self.label2id.get(word, UNK_TOKEN) for word in sentence]

    def decode_labels(self, sentence_ids):
        return [self.id2label[id] for id in sentence_ids]
    
    def mapping_seq(self, seqs, special_token = False):
        vocab = {}

        vocab['<PAD>'] = PAD_TOKEN

        if special_token:
            vocab['<UNK>'] = UNK_TOKEN


        for seq in seqs:
            for tok in seq: 
                if not vocab.get(tok):
                    vocab[tok] = len(vocab)

        return vocab
    
class Dataset(data.Dataset):
    def __init__(self, samples, lang):
        self.samples = samples
        self.lang = lang
        self.first = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence, score, labels = self.samples[idx]
        encoded_sentence = self.lang.encode_sent(sentence)
        encoded_labels = self.lang.encode_labels(labels)

        tensor_sentence = torch.LongTensor(encoded_sentence)
        tensor_labels = torch.LongTensor(encoded_labels)
        
        if self.first and INFO_ENABLED:
            print('- Sample')
            print('-- Sentence:', sentence)
            print('-- Encoded :', encoded_sentence)
            print('-- Label:', labels)
            print('-- Encoded:', encoded_labels)
            print('-- Score:', score)
            self.first = False


        return {'text':tensor_sentence, 'label':tensor_labels, 'score':score}
    

def collate_fn(batch):

    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        for i,l in enumerate(lengths):
            if l == 0:
                print(sequences[i])
                exit(0)
        max_len = 1 if max(lengths)==0 else max(lengths)

        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix

        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    new_item = {}
    for key in batch[0].keys():
        new_item[key] = [el[key] for el in batch]

    source, lengths = merge(new_item['text'])
    labels, _ = merge(new_item['label'])

    new_item['text'] = source.to(DEVICE)
    new_item['labels'] = labels.to(DEVICE)
    new_item['vader'] = torch.Tensor(new_item['score']).to(DEVICE)
    new_item['lengths'] = torch.LongTensor(lengths).to(DEVICE)
    if INFO_ENABLED:
        print('COLLATEFN:',new_item['text'].shape, new_item['labels'].shape, new_item['lengths'].shape, new_item['vader'].shape)
    return new_item