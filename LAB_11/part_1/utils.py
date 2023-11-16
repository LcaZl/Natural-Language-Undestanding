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
from nltk.lm.preprocessing import flatten
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
from tqdm import tqdm
# Parameters
PAD_TOKEN = 0
UNK_TOKEN = 1
DEVICE = 'cuda:0'
INFO_ENABLED = False
MAX_VOCAB_SIZE = 10000

def preprocess(dataset, label, mark_neg = True, file_id = 0):
    new_dataset = []
    for sent in dataset:
        text = ' '.join(sent)

        vscore = sia.polarity_scores(text)['compound']
        if vscore <= -0.6:
            vscore = 'VNEG'  # Molto negativo
        elif vscore <= -0.2:
            vscore = 'NEG'  # Negativo
        elif vscore <= 0.2:
            vscore = 'NET'  # Neutrale
        elif vscore <= 0.6:
            vscore = 'POS'  # Positivo
        else:
            vscore = 'VPOS'  # Molto positivo

        #stop_words = set(stopwords.words('english'))
        #lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        #tokens = [lemmatizer.lemmatize(word) for word in tokens]
        #tokens = [word for word in tokens if word not in stop_words]

        if mark_neg:
            tokens = mark_negation(tokens)

        if len(tokens) != 0:
            new_dataset.append((tokens, vscore, label, file_id))
        
    return new_dataset

def filter_movie_reviews(filter):
    mr = movie_reviews
    new_mr = {}
    categories = mr.categories()

    filter = set([' '.join(sentence) for sentence in filter])

    for category in categories:  # categories è una lista di categorie, ad es. ['neg', 'pos']
        for fileid in tqdm(movie_reviews.fileids(categories=category)):
            # Unire tutte le frasi di un documento in una singola lista di token
            processed_doc = preprocess(movie_reviews.sents(fileid), label='neg', mark_neg=True, file_id=fileid)                
            new_doc = []

            new_doc = [tokens for tokens, vscore, label, doc_file_id in processed_doc if ' '.join(tokens) not in filter]
            # Unire le frasi rimanenti per creare un nuovo documento
            if new_doc:
                new_mr[fileid] = [tok for sent in new_doc for tok in sent]

    return new_mr
    
def load_dataset(dataset_name, kfold, test_size = 0.1, args = []):
    print(f'Loading Dataset {dataset_name}...')

    if dataset_name == 'Subjectivity':

    # Load subjectivity dataset and preprocess texts
        print(' - Categories:', subjectivity.categories())
        grp1_sentences = preprocess(subjectivity.sents(categories='subj'), label='subj') # (Lista token, label)
        grp2_sentences = preprocess(subjectivity.sents(categories='obj'), label='obj') # (Lista token, label)
        categories = subjectivity.categories()

    elif dataset_name == 'Movie_reviews':

        mr = movie_reviews
        categories = mr.categories()
        print(' - Categories:', mr.categories())
        grp1_sentences = preprocess([list(flatten(doc)) for doc in mr.paras(categories='neg')], label = 'neg')
        grp2_sentences = preprocess([list(flatten(doc)) for doc in mr.paras(categories='pos')], label = 'pos')

    elif dataset_name == 'movie_review_4subjectivity': # List of all sent from movie review to be filtered

        mr = movie_reviews
        categories = mr.categories()

        all_sentences = []
    
        for file_id in tqdm(movie_reviews.fileids(categories='neg')):
            all_sentences.extend(preprocess(movie_reviews.sents(file_id), 'neg', file_id=file_id))
        for file_id in tqdm(movie_reviews.fileids(categories='pos')):
            all_sentences.extend(preprocess(movie_reviews.sents(file_id), 'pos', file_id=file_id))

        args[0].extend_classes(categories) # Lang
        dataset = Dataset(all_sentences, args[0])
        dataloader = DataLoader(dataset, batch_size = 128, collate_fn = collate_fn)
        return dataloader, None, args[0]
    
    elif dataset_name == 'movie_review_filtered':

        mr = filter_movie_reviews(args[0])
        categories = movie_reviews.categories()

        grp1_sentences = preprocess([mr[fileid] for fileid in movie_reviews.fileids(categories='pos')], label = 'pos')
        grp2_sentences = preprocess([mr[fileid] for fileid in movie_reviews.fileids(categories='neg')], label = 'neg')
    else:
        raise Exception('Dataset name not recognized.')
    
    all_sentences = grp1_sentences + grp2_sentences

    # Dividi il dataset in train e test
    train_sentences, test_sentences = train_test_split(all_sentences, test_size=test_size, random_state=42, shuffle = True)

    # Build vocabulary
    lang = Lang(train_sentences, categories)

    train_labels = [label for _, _, label, _ in train_sentences]
    fold_datasets = []  # This will store the dataset splits
    
    for k, (train_indices, val_indices) in enumerate(kfold.split(train_sentences, train_labels)):
        # Split the data into training and validation sets for the current fold
        train_samples = [train_sentences[i] for i in train_indices]
        val_samples = [train_sentences[i] for i in val_indices]

        print(f' - FOLD {k} - Train Size: {len(train_samples)} - Val Size: {len(val_samples)}')

        # Create Dataset instances
        train_dataset = Dataset(train_samples, lang)
        val_dataset = Dataset(val_samples, lang)

        train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, collate_fn = collate_fn)
        val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = True, collate_fn = collate_fn)
        
        # Store the datasets
        fold_datasets.append((train_loader, val_loader))

    test_dataset = Dataset(test_sentences, lang)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = True, collate_fn = collate_fn)

    # Subjectivity dataset
    #print(' - All sentences:', all_sentences[:3])
    print(' - Vocabulary size:', lang.vocab_size)
    print(' - Group ',grp1_sentences[0][1],' - First sent len:', len(grp1_sentences[0][0]), )
    print(' - Group ',grp2_sentences[0][1],' - First sent len:', len(grp2_sentences[0][0]), )
    print(f'{dataset_name} folds (', len(fold_datasets), '):')
    for k, fold in enumerate(fold_datasets):
        print('- Fold',k,' dim -> Train:',len(fold[0]), 'Dev:', len(fold[1]))

    print(' - Sample:', train_dataset[0])
    print('Datasets loaded!\n')
    return fold_datasets, test_loader, lang

class Lang:
    def __init__(self, text, classes):
        self.word2id = self.mapping_seq([el for el, _, _, _ in text], special_token = True)
        self.id2word = {id: word for word, id in self.word2id.items()}

        self.vlabel2id = self.mapping_seq([[vlabel for _, vlabel, _, _ in text]], special_token = False)
        self.id2vlabel = {id: vlabel for vlabel, id in self.vlabel2id.items()}

        self.vocab_size = len(self.word2id)

        self.class2id = {}
        for i, cls in enumerate(classes):
            self.class2id[cls] = i

        self.id2class = {i:c for c, i in self.class2id.items()}
        print(' - Vader label ids:',self.vlabel2id)
        print(' - Classes label ids:',self.class2id)

    def extend_classes(self, cls):
        for cl in cls:
            self.class2id[cl] = len(self.class2id)

        self.id2class = {i:c for c, i in self.class2id.items()}

    def encode(self, sentence):
        return [self.word2id.get(word, UNK_TOKEN) for word in sentence]

    def decode(self, sentence_ids):
        return [self.id2word[id] for id in sentence_ids]
    
    def mapping_seq(self, sentences, special_token = False):
        #word_counts = Counter(word for sent in sentences for word in sent)
        #most_common_words = word_counts.most_common(MAX_VOCAB_SIZE)

        vocab = {}
        vocab['<PAD>'] = PAD_TOKEN

        if special_token:
            vocab['<UNK>'] = UNK_TOKEN
        for sent in sentences:
            for word in sent: 
                if not vocab.get(word):
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
        sentence, vlabel, label, doc_id = self.samples[idx]
        encoded_sentence = self.lang.encode(sentence)

        tensor_sentence = torch.LongTensor(encoded_sentence)
        tensor_vlabel = self.lang.vlabel2id[vlabel]
        tensor_label = self.lang.class2id[label]

        if self.first and INFO_ENABLED:
            print('- Sample (Label:', label, ')')
            print('-- Sentence:', sentence)
            print('-- Encoded :', encoded_sentence)
            print('-- Label:', label)
            print('-- Encoded:', tensor_label)
            print('-- vlabel:', vlabel)
            print('--Encoded:', self.lang.vlabel2id[vlabel])
            self.first = False


        return {'text':tensor_sentence, 'vlabel': tensor_vlabel, 'label':tensor_label, 'docid': doc_id}
    
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
    new_item['vlabels'] = torch.LongTensor(new_item['vlabel']).to(DEVICE)

    #if INFO_ENABLED:
        #print('COLLATEFN:',new_item['text'].shape, new_item['vlabels'].shape, new_item['labels'].shape, new_item['lengths'].shape)
    return new_item

