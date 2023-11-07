import nltk
from nltk.sentiment.util import mark_negation
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import movie_reviews
import torch.utils.data as data
from sklearn.model_selection import KFold
# Assure that we have the necessary data downloaded
nltk.download('subjectivity')
from nltk.corpus import subjectivity
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('movie_reviews')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants
vlex = VaderConstants()

# Parameters
PAD_TOKEN = 0
UNK_TOKEN = 1
DEVICE = 'cuda:0'
INFO_ENABLED = False
MAX_VOCAB_SIZE = 10000

def preprocess(text, mark_neg = False):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    #tokens = [lemmatizer.lemmatize(word) for word in tokens]
    if mark_neg:
        tokens = mark_negation(tokens)

    return tokens

def load_dataset(dataset_name, kfold, test_size = 0.1, args = []):
    print(f'Loading Dataset {dataset_name}...')

    if dataset_name == 'Subjectivity':

    # Load subjectivity dataset and preprocess texts
        print(' - Categories:', subjectivity.categories())
        grp1_sentences = [(preprocess(' '.join(sent)), 'subj') for sent in subjectivity.sents(categories='subj')] # (Lista token, label)
        grp2_sentences = [(preprocess(' '.join(sent)), 'obj') for sent in subjectivity.sents(categories='obj')]
        categories = subjectivity.categories()

    elif dataset_name == 'Movie_reviews':

        mr = movie_reviews
        categories = mr.categories()

        print(' - Categories:', mr.categories())
        grp1_sentences = [(preprocess(' '.join([' '.join(sublist) for sublist in para]), mark_neg=True), 'neg') for para in mr.paras(categories='neg')]
        grp2_sentences = [(preprocess(' '.join([' '.join(sublist) for sublist in para])), 'pos') for para in mr.paras(categories='pos')]

    elif dataset_name == 'Filtered_movie_reviews':
        mr = movie_reviews
        categories = mr.categories()
        grp1_sentences = [(el, label) for el, label in args[0] if label == 'neg']
        grp2_sentences = [(el, label) for el, label in args[0] if label == 'pos']
    else:
        raise Exception('Dataset name not recognized.')
    
    all_sentences = grp1_sentences + grp2_sentences

    # Dividi il dataset in train e test
    train_sentences, test_sentences = train_test_split(all_sentences, test_size=test_size, random_state=42, shuffle = True)

    # Build vocabulary
    lang = Lang(train_sentences, categories)

    train_labels = [label for _, label in train_sentences]
    fold_datasets = []  # This will store the dataset splits
    for k, (train_indices, val_indices) in enumerate(kfold.split(train_sentences, train_labels)):
        # Split the data into training and validation sets for the current fold
        train_samples = [train_sentences[i] for i in train_indices]
        val_samples = [train_sentences[i] for i in val_indices]

        print(f' - FOLD {k} - Train Size: {len(train_samples)} - Val Size: {len(val_samples)}')

        # Create Dataset instances
        train_dataset = Dataset(train_samples, lang)
        val_dataset = Dataset(val_samples, lang)

        train_loader = DataLoader(train_dataset, batch_size = 100, shuffle = True, collate_fn = collate_fn)
        val_loader = DataLoader(val_dataset, batch_size = 100, shuffle = True, collate_fn = collate_fn)
        
        # Store the datasets
        fold_datasets.append((train_loader, val_loader))

    test_dataset = Dataset(test_sentences, lang)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True, collate_fn = collate_fn)

    # Subjectivity dataset
    #print(' - All sentences:', all_sentences[:3])
    print(' - Vocabulary size:', lang.vocab_size)
    print(' - Group ',grp1_sentences[0][1],' - First sent len:', len(grp1_sentences[0][0]), )
    print(' - Group ',grp2_sentences[0][1],' - First sent len:', len(grp2_sentences[0][0]), )
    print(f'{dataset_name} folds (', len(fold_datasets), '):')
    for k, fold in enumerate(fold_datasets):
        print('- Fold',k,' dim -> Train:',len(fold[0]), 'Dev:', len(fold[1]))
    print('Datasets loaded!\n')
    return fold_datasets, test_loader, lang

class Lang:
    def __init__(self, text, classes):
        self.word2id = self.mapping_seq([el for el, _ in text], special_token = True)
        self.vocab_size = len(self.word2id)
        self.id2word = {id: word for word, id in self.word2id.items()}
        self.class2id = {}
        for i, cls in enumerate(classes):
            self.class2id[cls] = i
        self.id2class = {i:c for c, i in self.class2id.items()}
        print(' - Classes Ids:',self.class2id)

    def encode(self, sentence):
        return [self.word2id.get(word, UNK_TOKEN) for word in sentence]

    def decode(self, sentence_ids):
        return [self.id2word[id] for id in sentence_ids]
    
    def mapping_seq(self, sentences, special_token = False):
        word_counts = Counter(word for sent in sentences for word in sent)
        most_common_words = word_counts.most_common(MAX_VOCAB_SIZE - 2 if special_token else MAX_VOCAB_SIZE)
        vocab = {}

        if special_token:
            vocab['<PAD>'] = PAD_TOKEN
            vocab['<UNK>'] = UNK_TOKEN

        for _, (word, _) in enumerate(most_common_words): 
            vocab[word] = len(vocab)

        return vocab

class Dataset(data.Dataset):
    def __init__(self, samples, lang):
        self.samples = samples
        self.lang = lang
        self.first = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence, label = self.samples[idx]
        encoded_sentence = self.lang.encode(sentence)

        tensor_sentence = torch.LongTensor(encoded_sentence)
        tensor_label = self.lang.class2id[label]

        if self.first and INFO_ENABLED:
            print('- Sample (Label:', label, ')')
            print('-- Sentence:', sentence)
            print('-- Encoded :', encoded_sentence)
            print('-- Label:', label)
            print('-- Encoded:', tensor_label)
            self.first = False


        return {'text':tensor_sentence, 'label':tensor_label}
    
# Preprocessing function

    

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

    new_item['text'] = source.to(DEVICE)
    new_item['labels'] = torch.LongTensor(new_item['label']).to(DEVICE)
    new_item['lengths'] = torch.LongTensor(lengths).to(DEVICE)
    #if INFO_ENABLED:
       # print('COLLATEFN:',new_item['text'].shape, new_item['labels'].shape, new_item['lengths'].shape)
    return new_item

