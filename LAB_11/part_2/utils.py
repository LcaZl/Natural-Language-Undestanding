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
from nltk.sentiment.util import mark_negation
from transformers import BertTokenizer

nltk.download('vader_lexicon')
nltk.download('sentiwordnet')
import torch
import en_core_web_sm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()
# Parameters
PAD_TOKEN = 0
UNK_TOKEN = 1
DEVICE = 'cpu'
TRAIN_PATH = 'dataset/laptop14_train.txt'
TEST_PATH = 'dataset/laptop14_test.txt'
INFO_ENABLED = True
BERT_MAX_LEN = 512

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

            aspect_tags = []
            pol_tags = []
            text = []
            is_in_aspect = False
            aspect_start_index = -1

            for i, w in enumerate(words_tagged):
                word, tag = w.rsplit('=', 1)

                if tag != 'O' and tag != 'ASPECT0' and tag != '':
                    aspect_tag, pol_tag = tag.split('-')
                    if aspect_tag == 'T':
                        if not is_in_aspect:
                            is_in_aspect = True
                            aspect_start_index = i
                            aspect_tags.append('B')  # Inizio di un aspetto
                        else:
                            aspect_tags.append('I')  # Continuazione di un aspetto
                    else:
                        if is_in_aspect:
                            # Fine dell'aspetto corrente
                            if aspect_start_index == i - 1:
                                aspect_tags[i] = 'S'  # Singolo termine di aspetto
                            else:
                                aspect_tags[i] = 'E'  # Fine di un aspetto multi-token
                            pol_tags.append((aspect_start_index, i-1, pol_tag))
                            is_in_aspect = False
                        aspect_tags.append('O')
                else:
                    if is_in_aspect:
                        # Fine dell'aspetto corrente
                        if aspect_start_index == i - 1:
                            aspect_tags[-1] = 'S'
                        else:
                            aspect_tags[-1] = 'E'
                        pol_tags.append((aspect_start_index, i-1, pol_tag))  # Presumiamo NEU se non specificato
                        is_in_aspect = False
                    aspect_tags.append('O')

                text.append(word)

            # Controlla se il sample finisce con un aspetto
            if is_in_aspect:
                if aspect_start_index == len(words_tagged) - 1:
                    aspect_tags[-1] = 'S'
                else:
                    aspect_tags[-1] = 'E'
                pol_tags.append((aspect_start_index, len(words_tagged) - 1, pol_tag))  # Presumiamo NEU se non specificato

            text = ' '.join(text)
            new_dataset.append((text, aspect_tags, pol_tags))
            if not INFO_ENABLED:
                print('- Raw       :', words_tagged)
                print('- Text      :', text)
                print('- Aspects   :', aspect_tags)
                print('- Polarities:', pol_tags, '\n')    
    return new_dataset



def load_dataset():
    print(f'\nLoading Dataset Laptop 14...')

    train_raw = read_file(TRAIN_PATH)
    test_raw = read_file(TEST_PATH)

    train_set = process_raw_data(train_raw)
    test_set = process_raw_data(test_raw)

    aspect_labels = [el[1] for el in train_set]
    pol_labels = [el[2] for el in train_set]

    lang = Lang(aspect_labels, pol_labels)

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

    print(' - Aspects labels :', lang.aspect2id)
    print(' - Polarity labels:', lang.pol2id)
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
    print(f' - Train dataset sample:', train_dataset[0])
    print(f' - Test sample:', test_set[0])
    print(f' - Test dataset sample:', test_dataset[0])
    print('Dataset loaded.\n\n')

    return fold_datasets, test_loader, lang
    
class Lang:
    def __init__(self, aspect_lbs, pol_lbs):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        self.aspect2id = self.mapping_seq(aspect_lbs, special_token = False)
        self.id2aspect = {id: label for label, id in self.aspect2id.items()}

        self.pol2id = self.mapping_seq(pol_lbs, special_token = False)
        self.id2pol = {id: label for label, id in self.pol2id.items()}

        self.vocab_size = len(self.tokenizer.vocab)
        self.aspects = len(self.aspect2id)
        self.polarities = len(self.pol2id)
    
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
    
    def map_aspect_ote(self, asp):
        if asp == 'T':
            return 'S'
        elif self.id2aspect[asp] == 'T':
            return 'S'
        
class Dataset(data.Dataset):
    def __init__(self, dataset, lang):
        self.lang = lang
        self.utt_ids, self.asp_ids, self.pol_ids, self.attention_masks, self.token_types = self.prepare_data(dataset)
        self.first = True

    def prepare_data(self, dataset):

        utt_ids = []
        asp_ids = []
        pol_ids = []
        attention_masks = []
        token_types = []

        for i, entry in enumerate(dataset):
            # Tokenization
            tokenized_entry = self.lang.tokenizer(entry[0])
            input_ids = tokenized_entry['input_ids']
            aligned_asp_ids, aligned_pol_ids = self.align_tags(entry[1], entry[2], entry[0].split(), input_ids)

            if INFO_ENABLED:
                print('----------------------------- Sample ', i, '-----------------------------')
                print('- Sent          :', entry[0])
                print('- Aspects       :', entry[1])
                print('- Polarities    :', entry[2])
                print('- Sent encoded  :', input_ids)
                print('- Asp. encoded  :', aligned_asp_ids)
                print('- Pol. encoded  :', aligned_pol_ids)
                print('- Token type ids:', tokenized_entry['token_type_ids'])
                print('- Attention mask:', tokenized_entry['attention_mask'])


            utt_ids.append(input_ids)
            asp_ids.append(aligned_asp_ids)
            pol_ids.append(aligned_pol_ids)
            attention_masks.append(tokenized_entry['attention_mask'])
            token_types.append(tokenized_entry['token_type_ids'])

            # Verify dimensionality
            assert len(input_ids) == len(aligned_asp_ids) == len(tokenized_entry['attention_mask']) == len(tokenized_entry['token_type_ids'])
            assert input_ids[0] == self.lang.cls_token_id and input_ids[-1] == self.lang.sep_token_id
            for pol in aligned_pol_ids:
                if pol[0] == pol[1]:
                    assert aligned_asp_ids[pol[0]] == 'S' and aligned_asp_ids[pol[1]] == 'S'
                else:
                    assert pol[0] < pol[1]
                    assert aligned_asp_ids[pol[0]] == 'B' and aligned_asp_ids[pol[1]] == 'E'
                    if pol[1] - pol[0] >> 1:
                        for asp_id in aligned_asp_ids[pol[0] + 1: pol[1] - 1]:
                            assert asp_id == 'I' 

        return utt_ids, asp_ids, pol_ids, attention_masks, token_types
    
    def align_tags(self, aspect_tags, pol_tags, words, input_ids):
        aligned_aspects = ['O'] * len(input_ids)  # Default 'O' for all tokens
        aligned_polarities = []
        current_aspect = 'O'
        aspect_start = None
        token_idx = 1  # Start from 1 to skip [CLS] token
        pol_idx = 0

        for word, aspect in zip(words, aspect_tags):
            sub_tokens = self.lang.tokenizer.tokenize(word)
            for sub_token in sub_tokens:
                
                if token_idx < len(input_ids) - 1:  # Skip [SEP] token
                    aligned_aspects[token_idx] = aspect
                    if aspect != 'O':
                        if current_aspect == 'O':  # Start of a new aspect
                            aspect_start = token_idx
                            aspect_sent = pol_tags[pol_idx][2]
                            pol_idx += 1                            
                        current_aspect = aspect
                    elif current_aspect != 'O':  # End of the current aspect
                        end_idx = token_idx - 1 if aspect_start != token_idx - 1 else aspect_start
                        aligned_polarities.append((aspect_start, end_idx, aspect_sent))
                        current_aspect = 'O'
                    token_idx += 1

        in_aspect = False
        for idx, asp in enumerate(aligned_aspects):
            if idx != len(aligned_aspects):
                if in_aspect and asp != 'E':
                    aligned_aspects[idx] = 'I'
                if asp == 'B':
                    in_aspect = True
                if asp == 'E':
                    in_aspect = False

        GESTIRE SSSSSSSSS

        
        # Check if the last aspect extends to the end
        if current_aspect != 'O':
            end_idx = token_idx - 1 if aspect_start != token_idx - 1 else aspect_start
            for pol_start, pol_end, sentiment in pol_tags:
                if pol_start == aspect_start and pol_end == end_idx:
                    aligned_polarities.append((aspect_start, end_idx, sentiment))

        return aligned_aspects, aligned_polarities
    
    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.utt_ids[idx]),
            'aspects': torch.tensor(self.asp_ids[idx]),
            'polarities': torch.tensor(self.pol_ids[idx]),
            'attention_mask': torch.tensor(self.attention_masks[idx]),
            'token_type_ids': torch.tensor(self.token_types[idx])
        }
    
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

    attention_mask, _ = merge(new_item['attention_mask'])
    token_type_ids, _ = merge(new_item['token_type_ids'])

    text = text.to(DEVICE)
    y_aspects = y_aspects.to(DEVICE)
    y_polarities = y_polarities.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)

    new_item["texts"] = text
    new_item["y_aspects"] = y_aspects
    new_item['y_polarities'] = y_polarities
    new_item["attention_mask"] = attention_mask
    new_item['token_type_ids'] = token_type_ids

    #sample = {'utterances': text.shape, 'yaspects':y_aspects.shape, 'ypolarities':y_aspects.shape,'attention_mask':attention_mask.shape}
    #print('-   Collate_fn :', sample)
    return new_item