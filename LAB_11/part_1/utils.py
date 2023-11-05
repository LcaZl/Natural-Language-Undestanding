import nltk
from nltk.sentiment.util import mark_negation
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from nltk.corpus import movie_reviews
import torch.utils.data as data
import itertools
from sklearn.model_selection import KFold
# Assure that we have the necessary data downloaded
nltk.download('subjectivity')
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import subjectivity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Parameters
PAD_TOKEN = 0
UNK_TOKEN = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_mr_dataset(kfold, vocab_size=10000, test_size = 0.2):
    print('\nLoading movie reviews dataset ...')

    MR = movie_reviews
    negative_rev = [(preprocess(' '.join([' '.join(sublist) for sublist in para]), mark_neg=True), 'neg') for para in MR.paras(categories='neg')]
    positive_rev = [(preprocess(' '.join([' '.join(sublist) for sublist in para])), 'pos') for para in MR.paras(categories='pos')]
    all_reviews = negative_rev + positive_rev

    # Dividi il dataset in train e test
    train_review, test_review = train_test_split(all_reviews, test_size=test_size, random_state=42)

    # Build vocabulary from preprocessed reviews
    vocab = build_vocab([review for review, label in train_review], vocab_size=vocab_size)
    lang = Lang(vocab)

    fold_datasets = []  # This will store the dataset splits
    for train_indices, val_indices in kfold.split(train_review):
        # Split the data into training and validation sets for the current fold
        train_samples = [train_review[i] for i in train_indices]
        val_samples = [train_review[i] for i in val_indices]

        # Create SubjectivityDataset instances
        train_dataset = SubjectivityDataset(train_samples, lang)
        val_dataset = SubjectivityDataset(val_samples, lang)

        # Store the datasets
        fold_datasets.append((train_dataset, val_dataset))

    test_dataset = SubjectivityDataset(test_review, lang)

    print(' - Vocabulary len:', len(vocab))
    print(' - Rev neg len:', len(negative_rev))
    print(' - Rev pos len:', len(positive_rev))
    print(' - Reviews len:', len(all_reviews))
    print(' - Movie Review sent:', all_reviews[0])
    print('Movie review datasets loaded!\n')

    return fold_datasets, test_dataset, lang

def load_subj_dataset(kfold, vocab_size=10000, test_size = 0.2):
    print('\nLoading subjectivity dataset ...')


    # Load subjectivity dataset and preprocess texts
    subj_sentences = [(preprocess(' '.join(sent)), 'subj') for sent in subjectivity.sents(categories='subj')]
    obj_sentences = [(preprocess(' '.join(sent)), 'obj') for sent in subjectivity.sents(categories='obj')]
    all_sentences = subj_sentences + obj_sentences

    # Dividi il dataset in train e test
    train_sentences, test_sentences = train_test_split(all_sentences, test_size=test_size, random_state=42)

    # Build vocabulary
    vocab = build_vocab([el for el, _ in train_sentences], vocab_size=vocab_size)
    lang = Lang(vocab)

    fold_datasets = []  # This will store the dataset splits
    for train_indices, val_indices in kfold.split(train_sentences):
        # Split the data into training and validation sets for the current fold
        train_samples = [train_sentences[i] for i in train_indices]
        val_samples = [train_sentences[i] for i in val_indices]

        # Create SubjectivityDataset instances
        train_dataset = SubjectivityDataset(train_samples, lang)
        val_dataset = SubjectivityDataset(val_samples, lang)

        # Store the datasets
        fold_datasets.append((train_dataset, val_dataset))

    test_dataset = SubjectivityDataset(test_sentences, lang)

    # Subjectivity dataset
    print(' - All sentences:', all_sentences[:3])
    print(' - Vocabulary size:', len(vocab))
    print(' - Subjective sents:', len(subjectivity.sents(categories='subj')))
    print(' - Objective sents:', len(subjectivity.sents(categories='obj')))
    print(' - Subjectivity sent:', subjectivity.sents(categories='subj')[0])
    print('Subjectivity datasets loaded!\n')

    return fold_datasets, test_dataset, lang

# Preprocessing function
def preprocess(text, mark_neg = False):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    if mark_neg:
        tokens = mark_negation(tokens)

    return tokens

# Function to build vocabulary
def build_vocab(sentences, vocab_size):
    word_counts = Counter(word for sent in sentences for word in sent)
    most_common_words = word_counts.most_common(vocab_size)
    vocab = {word: i+2 for i, (word, _) in enumerate(most_common_words)}  # start from 2 to account for PAD and UNK
    vocab['<PAD>'] = PAD_TOKEN
    vocab['<UNK>'] = UNK_TOKEN
    return vocab


class Lang:
    def __init__(self, vocab):
        self.vocab_size = len(vocab) - 1 # PAD ?
        self.word2id = vocab
        self.id2word = {id: word for word, id in vocab.items()}

    def encode(self, sentence):
        return [self.word2id.get(word, UNK_TOKEN) for word in sentence]

    def decode(self, sentence_ids):
        return [self.id2word[id] for id in sentence_ids]

class SubjectivityDataset(data.Dataset):
    def __init__(self, samples, lang):
        self.samples = samples
        self.lang = lang
        self.first = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence, label = self.samples[idx]
        encoded_sentence = self.lang.encode(sentence)

        if self.first:
            print('- Sample (Label:', label, ')')
            print('-- Sentence:', sentence)
            print('-- Encoded :', encoded_sentence)
            self.first = False

        tensor_sentence = torch.Tensor(encoded_sentence)
        tensor_label = torch.LongTensor([1] if label == 'subj' else [0])

        return {'sent':tensor_sentence, 'length': len(encoded_sentence), 'label':tensor_label}

class MovieReviewsDataset(data.Dataset):
    def __init__(self, samples, lang):
        self.samples = samples
        self.lang = lang
        self.first = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence, label = self.samples[idx]
        encoded_sentence = self.lang.encode(sentence)

        if self.first:
            print('- Sample (Label:', label, ')')
            print('-- Sentence:', sentence)
            print('-- Encoded :', encoded_sentence)
            self.first = False

        tensor_sentence = torch.Tensor(encoded_sentence)
        tensor_label = torch.LongTensor([1] if label == 'pos' else [0])

        return {'sent':tensor_sentence, 'length': len(encoded_sentence), 'label':tensor_label}
    
def collate_fn(batch):
    # Sort the batch in the descending order of sentence length
    batch.sort(key=lambda x: x['length'], reverse=True)

    # Extract sentences and labels from the sorted batch data
    samples = [item['sent'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)

    # Perform padding for the sorted sentences to reach the same length
    padded_sentences = pad_sequence(samples, batch_first=True, padding_value=PAD_TOKEN)

    # Convert samples to torch.LongTensor as they represent word indices
    padded_sentences = padded_sentences.long()

    # Return the tensors as a dictionary
    return {'sent': padded_sentences, 'label': labels, 'lengths': lengths}
