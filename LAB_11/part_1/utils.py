import nltk
import torch
import torch.utils.data as data
import math

from nltk.sentiment.util import mark_negation
from nltk.lm.preprocessing import flatten
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from tqdm import tqdm

nltk.download('subjectivity')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('vader_lexicon')
from nltk.corpus import subjectivity
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

PAD_TOKEN = 0
BERT_MAX_LEN = 512

DEVICE = 'cuda:0'
INFO_ENABLED = False

def preprocess(dataset, label, mark_neg = True, file_id = 0):
    new_dataset = []
    for tokens in tqdm(dataset, desc = 'Preprocessing dataset'):
        text = ' '.join(tokens)

        vscore = sia.polarity_scores(text)['compound']
        if vscore <= -0.5:
            vscore = 'VNEG'  # Molto negativo
        elif vscore <= 0:
            vscore = 'NEG'  # Negativo
        elif vscore <= 0.5 :
            vscore = 'POS'  # Positivo
        else:
            vscore = 'VPOS'  # Molto positivo

        stop_words = set(stopwords.words('english'))
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stop_words]

        if mark_neg:
            tokens = mark_negation(tokens)

        if len(tokens) != 0:
            new_dataset.append((tokens, label, file_id))
        
    return new_dataset

def filter_movie_reviews(filter):
    mr = movie_reviews
    new_mr = {}
    categories = mr.categories()

    filter = set([' '.join(sentence) for sentence in filter])

    for category in categories: # ['neg','pos']
        for fileid in tqdm(movie_reviews.fileids(categories=category)):

            # Document from list of list to a single list of tokens
            processed_doc = preprocess(movie_reviews.sents(fileid), label='neg', mark_neg=True, file_id=fileid)                

            new_doc = [tokens for tokens, _, _, _ in processed_doc if ' '.join(tokens) not in filter]
            
            # Creating new filtered document
            if new_doc:
                new_mr[fileid] = [tok for sent in new_doc for tok in sent]

    return new_mr
    
def load_dataset(dataset_name, kfold, test_size = 0.1, args = []):
    print(f'\nLoading Dataset {dataset_name}...')

    if dataset_name == 'Subjectivity':

        categories = subjectivity.categories()
        print(' - Categories:', categories)
        grp1_sentences = preprocess(subjectivity.sents(categories='subj'), label='subj') # (Lista token, label)
        grp2_sentences = preprocess(subjectivity.sents(categories='obj'), label='obj') # (Lista token, label)

    elif dataset_name == 'Movie_reviews':

        mr = movie_reviews
        categories = mr.categories()
        print(' - Categories:', categories)
        grp1_sentences = preprocess([list(flatten(doc)) for doc in mr.paras(categories='neg')], label = 'neg')
        grp2_sentences = preprocess([list(flatten(doc)) for doc in mr.paras(categories='pos')], label = 'pos')

    # Mr is a list od doc (list of list of list). Here is transformed into list of sentences, to feed the subjectivity model.
    # These sentences will be then filtered.
    elif dataset_name == 'movie_review_4subjectivity':

        mr = movie_reviews
        categories = mr.categories()

        # We only need the sentences, no other information and no distinction.
        all_sentences = []

        for file_id in tqdm(movie_reviews.fileids(categories='neg')):
            all_sentences.extend(preprocess(movie_reviews.sents(file_id), 'neg', file_id=file_id))
        for file_id in tqdm(movie_reviews.fileids(categories='pos')):
            all_sentences.extend(preprocess(movie_reviews.sents(file_id), 'pos', file_id=file_id))


        lang = Lang(categories)
        dataset = Dataset(all_sentences, args[0])
        dataloader = DataLoader(dataset, batch_size = 64, collate_fn = collate_fn)

        return dataloader, None, lang
    
    elif dataset_name == 'movie_review_filtered':

        mr = filter_movie_reviews(args[0]) # args[0] contains the sentences to remove (is the filter).
        categories = movie_reviews.categories()

        # Create standard dataset
        grp1_sentences = preprocess([mr[fileid] for fileid in movie_reviews.fileids(categories='pos')], label = 'pos')
        grp2_sentences = preprocess([mr[fileid] for fileid in movie_reviews.fileids(categories='neg')], label = 'neg')

    else:
        raise Exception('Dataset name not recognized.')
    
    all_sentences = grp1_sentences + grp2_sentences
    train_sentences, test_sentences = train_test_split(all_sentences, test_size=test_size, random_state=42, shuffle = True)

    lang = Lang(categories)

    train_labels = []
    for tokens, label, _ in train_sentences:
        num_segments = math.ceil(len(tokens) / BERT_MAX_LEN)
        train_labels.extend([label] * num_segments)    

    fold_datasets = []
    
    for k, (train_indices, val_indices) in enumerate(kfold.split(train_sentences, train_labels)):

        train_samples = [train_sentences[i] for i in train_indices]
        val_samples = [train_sentences[i] for i in val_indices]

        print(f' - FOLD {k} - Train Size: {len(train_samples)} - Val Size: {len(val_samples)}')

        train_dataset = Dataset(train_samples, lang)
        val_dataset = Dataset(val_samples, lang)
        train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, collate_fn = collate_fn)
        val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True, collate_fn = collate_fn)
        fold_datasets.append((train_loader, val_loader))

    print(f' - TEST SET - Size: {len(test_sentences)}')
    test_dataset = Dataset(test_sentences, lang)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True, collate_fn = collate_fn)

    # Info
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
    def __init__(self, classes):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = len(self.tokenizer.vocab)

        self.class2id = {}
        for i, cls in enumerate(classes):
            self.class2id[cls] = i

        self.id2class = {i:c for c, i in self.class2id.items()}
        print(' - Classes label ids:',self.class2id)

    def encode(self, sentence):
        return self.tokenizer.encode(sentence)

    def decode(self, sentence_ids):
        return self.tokenizer.decode(sentence_ids)

class Dataset(data.Dataset):
    def __init__(self, samples, lang):
        self.samples = []
        self.lang = lang
        self.first = True

        for sentence, label, doc_id in tqdm(samples, desc='Chuncking samples'):
            tokenized_sent = self.lang.tokenizer(' '.join(sentence), truncation=False, padding=False)
            encoded_sentence = tokenized_sent['input_ids']
            attention_mask = tokenized_sent['attention_mask']

            chunked_sentences = self.chunk_sequence(encoded_sentence)
            chunked_attention_masks = self.chunk_sequence(attention_mask)

            for sent, mask in zip(chunked_sentences, chunked_attention_masks):
                self.samples.append((sent, mask, label, doc_id))

    def __len__(self):
        return len(self.samples)

    def chunk_sequence(self, sequence):
        return [sequence[i:i + BERT_MAX_LEN] for i in range(0, len(sequence), BERT_MAX_LEN)]

    def __getitem__(self, idx):
        sentence, mask, label, doc_id = self.samples[idx]

        attention_mask = torch.tensor(mask)
        tensor_sentence = torch.tensor(sentence)
        tensor_label = self.lang.class2id[label]

        if self.first and INFO_ENABLED:
            print('- Sample (Label:', label, ')')
            print('-- Sentence:', sentence)
            print('-- Label:', label)
            print('-- Encoded:', tensor_label)
            print('--attMask:', attention_mask)
            self.first = False


        return {'text':tensor_sentence, 'attention_mask':attention_mask, 'label':tensor_label, 'docid': doc_id}
# Preprocessing function
def collate_fn(batch):

    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [min(len(seq), BERT_MAX_LEN) for seq in sequences]

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

    source, _ = merge(new_item['text'])
    attention_masks, _ = merge(new_item['attention_mask'])

    new_item['text'] = source.to(DEVICE)
    new_item['labels'] = torch.tensor(new_item['label']).to(DEVICE)
    new_item['attention_masks'] = attention_masks.to(DEVICE)
    #if INFO_ENABLED:
        #print('COLLATEFN:',new_item['text'].shape.shape, new_item['labels'].shape, new_item['lengths'].shape) # , new_item['vlabels']
    return new_item

