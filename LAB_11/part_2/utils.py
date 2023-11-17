# Add functions or classes used for data loading and preprocessing
import string
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.util import mark_negation
from sklearn.model_selection import StratifiedKFold
from nltk.sentiment import SentimentIntensityAnalyzer
import torch.utils.data as data
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.sentiment.util import mark_negation
from transformers import BertTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('vader_lexicon')
nltk.download('sentiwordnet')
import torch
import en_core_web_sm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()
# Parameters
PAD_TOKEN = 0

DEVICE = 'cuda:0'
TRAIN_PATH = 'dataset/laptop14_train.txt'
TEST_PATH = 'dataset/laptop14_test.txt'
INFO_ENABLED = False
BERT_MAX_LEN = 512

PRINTABLE = 3

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = file.read()
    dataset = dataset.split('\n')
    dataset = [el.strip() for el in dataset]
    return dataset

def process_raw_data(dataset):
    new_dataset = []
    lemmatizer = WordNetLemmatizer()

    for sample in dataset:

        if sample: 

            _, words_tagged = sample.split('####')
            words_tagged = words_tagged.split()

            text, aspect_tags, pol_tags = [], [], []
            is_in_aspect = False
            aspect_start_index = -1

            for i, w in enumerate(words_tagged):

                word, tag = w.rsplit('=', 1)
                text.append(word)

                if tag != 'O' and tag != 'ASPECT0' and tag != '':

                    _, pol_tag = tag.split('-')
                    if not is_in_aspect:
                        is_in_aspect = True
                        aspect_start_index = i
                        aspect_tags.append('S')  # Inizio di un aspetto
                    else:
                        aspect_tags.append('S')  # Continuazione di un aspetto
                else:

                    if is_in_aspect:
                        # Fine dell'aspetto corrente
                        aspect_tags[-1] = 'S'
                        pol_tags.append((aspect_start_index, i-1, pol_tag))
                        is_in_aspect = False
                    aspect_tags.append('O')

            # Controlla se il sample finisce con un aspetto
            if is_in_aspect:
                aspect_tags[-1] = 'S'
                pol_tags.append((aspect_start_index, len(words_tagged) - 1, pol_tag))

            #text = [word.lower() for word in text if word.isalpha()]
            #text = [lemmatizer.lemmatize(word) for word in text]
            new_dataset.append((' '.join(text), aspect_tags, pol_tags))

            if INFO_ENABLED:
                print('- Raw       :', words_tagged)
                print('- Text      :', text)
                print('- Aspects   :', aspect_tags)
                print('- Polarities:', pol_tags, '\n')    

    return new_dataset
import math
def calculate_inverse_weights(frequencies, lang):
    total_count = sum(frequencies.values())
    weights = {label: total_count / (freq + math.e) for label, freq in frequencies.items()}
    total_weight = sum(weights.values())
    normalized_weights = {label: round(weight / total_weight, 3) for label, weight in weights.items()}
    return normalized_weights

def load_dataset(skf):

    print(f'\nLoading Dataset Laptop 14...')

    train_raw = read_file(TRAIN_PATH)
    test_raw = read_file(TEST_PATH)

    train_set = process_raw_data(train_raw)
    test_set = process_raw_data(test_raw)

    lang = Lang()

    fold_datasets = []  # This will store the dataset splits
    stratify_labels = []

    for _, _, pol_tags in train_set:
        v = len(pol_tags)
        if v == 0 or v == 1 or v == 2 or v == 3:
            stratify_labels.append(v)
        elif len(pol_tags) > 3:
            stratify_labels.append(4)
    count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    for l in stratify_labels:
        count[l] += 1

    #print(count)

    for k, (train_indices, val_indices) in enumerate(skf.split(train_set, stratify_labels)):

        train_samples = [train_set[idx] for idx in train_indices]
        val_samples = [train_set[idx] for idx in val_indices]

        print(f'\n - FOLD {k} - Train Size: {len(train_samples)} - Val Size: {len(val_samples)}')

        train_dataset = Dataset(train_samples, lang)
        val_dataset = Dataset(val_samples, lang)

        # Inizializza i dizionari per il conteggio delle frequenze
        aspect_frequencies = {id : 0 for id, asp in lang.id2aspect.items()}
        polarity_frequencies = {id : 0 for id, asp in lang.id2pol.items()}

        # Aggiorna il conteggio per ogni campione nel dataset
        for aspect_tags, pol_tags in zip(train_dataset.asp_ids, train_dataset.pol_ids):
            for aspect in aspect_tags:
                if aspect in aspect_frequencies:
                    aspect_frequencies[aspect] += 1
            for pol in pol_tags:
                if pol in polarity_frequencies:
                    polarity_frequencies[pol] += 1

        aspect_weights = calculate_inverse_weights(aspect_frequencies, lang)
        polarity_weights = calculate_inverse_weights(polarity_frequencies, lang)
        print(' - Aspects frequencies:', aspect_frequencies, '\n - Aspects weigth:', aspect_weights) 
        print(' - Polarities frequencies:', polarity_frequencies, '\n - Polarities weigth:', polarity_weights) 

        train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, collate_fn = collate_fn)
        val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True, collate_fn = collate_fn)
        
        fold_datasets.append((train_loader, val_loader, list(aspect_weights.values()), list(polarity_weights.values())))
        #print(aspect_weights_tensor, polarity_weights_tensor) 

    test_dataset = Dataset(test_set, lang)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True, collate_fn = collate_fn)

    print(' - Aspects labels :', lang.aspect2id)
    print(' - Polarity labels :', lang.pol2id)
    print(' - Vocabulary size:', lang.vocab_size)
    print(' - Special tokens (CLS e SEP ids):', lang.cls_token_id, lang.sep_token_id)
    print(' - Raw sent:', train_raw[0])
    print(' - Raw training samples:', len(train_raw))
    print(' - Preprocessed training len:', len(train_set))
    print(' - Training dataset:', len(train_dataset))
    print(' - Test sents:', len(test_raw))
    print(f' - Test len:', len(test_set))
    print(' - Test dataset:', len(test_dataset))
    print(f' - Training len:', len(train_set))
    print(f' - Train sample:', train_set[0])
    print(f' - Test sample:', test_set[0])
    print('Dataset loaded.\n\n')

    return fold_datasets, test_loader, lang
    
class Lang:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        self.aspect2id = {'O':0, 'B':1, 'I':2, 'E':3, 'S':4}
        self.id2aspect = {id: label for label, id in self.aspect2id.items()}
        self.aspect_labels = len(self.aspect2id)

        self.pol2id = {'O':0 ,'POS': 1, 'NEG': 2, 'NEU': 3}
        self.id2pol = {id: label for label, id in self.pol2id.items()}
        self.polarity_labels = len(self.pol2id)

        self.vocab_size = len(self.tokenizer.vocab)
        
    def decode_aspects(self, aspects):
        decoded_asp = []
        for aspect in aspects:
            decoded_asp.append(self.id2aspect[aspect])
        return decoded_asp
    
    def decode_polarities(self, polarities):
        decoded_pol = []
        for pol in polarities:
            decoded_pol.append(self.id2pol[pol])
        return decoded_pol
    
    def encode_asppol(self, data):
        encoded_seq = []
        for tok in data:
            sp_tok = tok.split('-')
            if len(sp_tok) == 2:
                asp, pol = sp_tok
                id = int(str(self.aspect2id[asp]) + str(self.pol2id[pol]))
                encoded_seq.append(id)
            else:
                asp, pol = 'O', 'O'
                encoded_seq.append(self.aspect2id[asp])
        return encoded_seq
    
    def decode_asppol(self, data):
        decoded_seq = []

        for tok in data:
            if tok != -1:
                tok = str(tok)
                if len(tok) == 2:
                    asp = self.id2aspect[int(tok[0])]
                    pol = self.id2pol[int(tok[1])]
                    decoded_seq.append(f'{asp}-{pol}')
                else:
                    decoded_seq.append('O')
        return decoded_seq
    
class Dataset(data.Dataset):
    def __init__(self, dataset, lang):
        self.lang = lang
        self.utt_ids, self.asp_ids, self.pol_ids, self.asp_pol_ids, self.asp_pol_indexes, self.attention_masks, self.token_types = self.prepare_data(dataset)
        self.first = True
        self.print_sample = 0

    def prepare_data(self, dataset):

        utt_ids = []
        asp_ids = []
        pol_ids = []
        asp_pol_ids = []
        asp_pol_indexes = []
        attention_masks = []
        token_types = []

        for i, entry in enumerate(dataset):

            tokenized_entry = self.lang.tokenizer(entry[0])
            input_ids = tokenized_entry['input_ids']
        
            # Tokenization
            if INFO_ENABLED:
                print('----------------------------- Sample ', i, '-----------------------------')
                print('- Sent          :', entry[0].split())
                print('- Aspects       :', entry[1])
                print('- Polarities    :', entry[2])
                print('- Encoded sentencence     :', input_ids)

            aligned_aspect, aligned_polarity, aligned_asp_pol, asp_pol_index = self.align_tags(entry[1], entry[2], entry[0].split(), input_ids)

            if INFO_ENABLED:
                print('- Aligned encoded aspects :', aligned_aspect)
                print('- Aligned encoded Polarity:', aligned_polarity)
                print('- Aligned encoded Asp/Pol :', aligned_asp_pol)
                print('- Decoded Asp/Pol         :', self.lang.decode_asppol(aligned_asp_pol))
                print('- Asp/Pol indexes         :', asp_pol_index)
                print('- Token type ids          :', tokenized_entry['token_type_ids'])
                print('- Attention mask          :', tokenized_entry['attention_mask'])


            utt_ids.append(input_ids)
            asp_ids.append(aligned_aspect)
            pol_ids.append(aligned_polarity)
            asp_pol_ids.append(aligned_asp_pol)
            asp_pol_indexes.append(asp_pol_index)
            attention_masks.append(tokenized_entry['attention_mask'])
            token_types.append(tokenized_entry['token_type_ids'])

            # Verify sample
            assert len(input_ids) == len(aligned_aspect) == len(tokenized_entry['attention_mask']) == len(tokenized_entry['token_type_ids']) == len(aligned_polarity)
            assert input_ids[0] == self.lang.cls_token_id and input_ids[-1] == self.lang.sep_token_id
            for pol in asp_pol_index:
                if pol[0] == pol[1]:
                    assert aligned_aspect[pol[0]] == self.lang.aspect2id['S'] and aligned_aspect[pol[1]] == self.lang.aspect2id['S']
                else:
                    assert pol[0] < pol[1]
                    assert aligned_aspect[pol[0]] == self.lang.aspect2id['B'] and aligned_aspect[pol[1]] == self.lang.aspect2id['E']
                    if pol[1] - pol[0] >> 1:
                        for asp_id in aligned_aspect[pol[0] + 1: pol[1] - 1]:
                            assert asp_id == self.lang.aspect2id['I']

        return utt_ids, asp_ids, pol_ids, asp_pol_ids, asp_pol_indexes, attention_masks, token_types
    
    def align_tags(self, aspect_tags, pol_tags, words, input_ids):
        
        aligned_aspect = ['O'] * len(input_ids)  # Default 'O' for all tokens
        asp_pol_indexes = []

        current_aspect = 'O'
        aspect_start = None
        token_idx = 1  # Start from 1 to skip [CLS] token
        pol_idx = 0

        for word, aspect in zip(words, aspect_tags):
            sub_tokens = self.lang.tokenizer.tokenize(word)
            for _ in sub_tokens:
                
                if token_idx < len(input_ids) - 1:  # Skip [SEP] token
                    aligned_aspect[token_idx] = aspect
                    if aspect != 'O':
                        if current_aspect == 'O':  # Start of a new aspect
                            aspect_start = token_idx
                            aspect_sent = pol_tags[pol_idx][2]
                            pol_idx += 1                            
                        current_aspect = aspect
                    elif current_aspect != 'O':  # End of the current aspect
                        end_idx = token_idx - 1 if aspect_start != token_idx - 1 else aspect_start
                        asp_pol_indexes.append((aspect_start, end_idx, aspect_sent))
                        current_aspect = 'O'
                    token_idx += 1

        in_aspect = False
        for idx, asp in enumerate(aligned_aspect):
                if asp == 'S':
                    if not in_aspect and aligned_aspect[idx + 1] == 'S': 
                        in_aspect = True
                        aligned_aspect[idx] = self.lang.aspect2id['B']
                    elif in_aspect and aligned_aspect[idx + 1] == 'S':
                        aligned_aspect[idx] = self.lang.aspect2id['I']
                    elif in_aspect and not aligned_aspect[idx + 1] == 'S':
                        aligned_aspect[idx] = self.lang.aspect2id['E']
                        in_aspect = False
                    else:
                        aligned_aspect[idx] = self.lang.aspect2id[asp]
                else:
                    aligned_aspect[idx] = self.lang.aspect2id[asp]
                    
        if INFO_ENABLED:
            print('- Aligned aspecs          :', aligned_aspect)
            print('- Decoded al. en. Aspects :', self.lang.decode_aspects(aligned_aspect))

        aligned_asp_pol = self.lang.decode_aspects(aligned_aspect)
        aligned_polarity = [self.lang.pol2id['O']] * len(input_ids)

        for pol in asp_pol_indexes:
            for i in range(pol[0], pol[1] + 1):
                aligned_polarity[i] = self.lang.pol2id[pol[2]]
                aligned_asp_pol[i] = f'{aligned_asp_pol[i]}-{pol[2]}'

        return aligned_aspect, aligned_polarity, self.lang.encode_asppol(aligned_asp_pol), asp_pol_indexes
    
    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        if self.print_sample < PRINTABLE and INFO_ENABLED:
            print('----------------------------- Sample ', idx, '-----------------------------')
            print('- Sent Ids         :', self.utt_ids[idx])
            print('- Aspects ids      :', self.asp_ids[idx])
            print('- Polarities       :', self.pol_ids[idx])
            print('- asp_pol_ids      :', self.asp_pol_ids[idx])
            print('- asp_pol_ids_dec  :', self.lang.decode_asppol(self.asp_pol_ids[idx]))
            print('- Asp. Pol. Indexes:', self.asp_pol_indexes[idx])
            print('- Attention mask   :', self.attention_masks[idx])
            print('- Token type ids   :', self.token_types[idx])
            self.print_sample += 1
        sample =  {
            'text': torch.tensor(self.utt_ids[idx]),
            'aspects': torch.tensor(self.asp_ids[idx]),
            'polarities': torch.tensor(self.pol_ids[idx]),
            'asp_pol_indexes': self.asp_pol_indexes[idx],
            'asp_pol_ids': torch.tensor(self.asp_pol_ids[idx]),
            'attention_mask': torch.tensor(self.attention_masks[idx]),
            'token_type_ids': torch.tensor(self.token_types[idx])
        }

        return sample

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [min(len(seq), BERT_MAX_LEN) for seq in sequences]  # Capture effective lengths but ensure they don't exceed 512
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        
        for i, seq in enumerate(sequences):
            end = lengths[i]  # Use the effective length for padding
            padded_seqs[i, :end] = seq[:end]

        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x['text']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    text, y_lengths = merge(new_item['text'])
    y_aspects, _ = merge(new_item["aspects"]) 
    y_polarities, _ = merge(new_item["polarities"]) 
    y_asp_pol, _ = merge(new_item['asp_pol_ids'])

    attention_mask, _ = merge(new_item['attention_mask'])
    token_type_ids, _ = merge(new_item['token_type_ids'])

    text = text.to(DEVICE)
    y_aspects = y_aspects.to(DEVICE)
    y_polarities = y_polarities.to(DEVICE)
    y_asp_pol = y_asp_pol.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)

    new_item["texts"] = text
    new_item["y_aspects"] = y_aspects
    new_item['y_polarities'] = y_polarities
    new_item['y_asppol'] = y_asp_pol
    new_item["attention_mask"] = attention_mask
    new_item['token_type_ids'] = token_type_ids

    if INFO_ENABLED:
        sample = {'utterances': text.shape, 
                  'yaspects':y_aspects.shape, 
                  'ypolarities':y_aspects.shape,
                  'attention_mask':attention_mask.shape, 
                  'y_asppol':y_asp_pol.shape}
        print('-   Collate_fn :', sample)

    return new_item
