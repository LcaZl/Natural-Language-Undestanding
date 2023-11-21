# Add functions or classes used for data loading and preprocessing
import json
from pprint import pprint
from collections import Counter
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

PAD_TOKEN = 0
UNK_TOKEN = 1
BERT_MAX_LEN = 512
DEVICE = 'cuda:0'

def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def load_dataset():
    tmp_train_raw = load_data('../ATIS/train.json')
    test_raw = load_data('../ATIS/test.json')
    print('\nLoading and preparing dataset ...\n')

    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10)/(len(tmp_train_raw)),2)

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)
    Y = []
    X = []
    mini_Train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occure once only, we put them in training
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])
    # Random Stratify
    train_raw, dev_raw, y_train, y_dev = train_test_split(X, Y, test_size=portion,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=Y)
    train_raw.extend(mini_Train)
    corpus = train_raw

    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(intents, slots)

    print(f'Dataset info:')
    print(' - Train samples:', len(tmp_train_raw))
    print(' - Test samples:', len(test_raw))
    print(' - Dev samples:', len(dev_raw))
    print(f' - Portion: {portion}')
    
    # Create datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    return train_dataset, dev_dataset, test_dataset, lang

class Lang():
    def __init__(self, intents, slots):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        self.slot2id = self.lab2id(list(slots), include_special_token = True) 
        self.intent2id = self.lab2id(list(intents), include_special_token = True)
        
        self.id2intent = {v: k for k, v in self.intent2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}

    def lab2id(self, elements, include_special_token = False):
        """Convert list of labels to a dictionary mapping."""

        vocab = {}
        if (include_special_token):
            vocab['[PAD]'] = PAD_TOKEN
            vocab['[UNK]'] = UNK_TOKEN

        for elem in elements:
            
            vocab[elem] = len(vocab)

        return vocab

TESTING = True
class IntentsAndSlots(data.Dataset):

    def __init__(self, dataset, lang):
        self.lang = lang
        self.utt_ids, self.slot_ids, self.intent_ids, self.masks, self.token_type_ids = self.prepare_data(dataset)
        if TESTING:
            test_len = int(len(self.utt_ids)/4)
            self.utt_ids = self.utt_ids[:test_len]
            self.slot_ids = self.slot_ids[:test_len]
            self.intent_ids = self.intent_ids[:test_len] 
            self.masks = self.masks[:test_len] 
            self.token_type_ids = self.token_type_ids[:test_len]

    def prepare_data(self, dataset):

        utt_ids = []
        slot_ids = []
        intent_ids = []
        attention_masks = []
        token_types = []

        for i, entry in enumerate(dataset):

            # Tokenization
            tokenized_entry = self.lang.tokenizer(entry['utterance'])
            input_ids = tokenized_entry['input_ids']
            """         
            print('----------------------------- Sample ', i, '-----------------------------')
            print('- Entry - Phrase:', entry['utterance'])
            print('-     Intent    :', entry['intent'])
            print('-    input_ids  :', input_ids)
            print('- Attention mask:', tokenized_entry['attention_mask'])
            print('- Token type ids:',tokenized_entry['token_type_ids'])
            print('-      Slots    :', entry['slots'])
            """
            # Aligning slot labels with tokens
            aligned_slot_ids = self.align_slots(entry['slots'].split(), entry['utterance'].split())

            utt_ids.append(input_ids)
            slot_ids.append(aligned_slot_ids)
            intent_ids.append(self.lang.intent2id.get(entry['intent'], UNK_TOKEN))

            # Attention mask and token type ids
            attention_masks.append(tokenized_entry['attention_mask'])
            token_types.append(tokenized_entry['token_type_ids'])

            # Verify dimensionality
            assert len(input_ids) == len(aligned_slot_ids) == len(tokenized_entry['attention_mask']) == len(tokenized_entry['token_type_ids'])
            assert input_ids[0] == self.lang.cls_token_id and input_ids[-1] == self.lang.sep_token_id
            assert aligned_slot_ids[0] == self.lang.slot2id['O'] and aligned_slot_ids[-1] == self.lang.slot2id['O']

        return utt_ids, slot_ids, intent_ids, attention_masks, token_types

    
    def align_slots(self, slot_labels, utterance_words):
        aligned_slots = [self.lang.slot2id['O']]
        #bert_tokenize_phrase = [] # Only for print

        slot_pointer = 0
        
        for word in utterance_words:
            first = True    
            sub_tokens = self.lang.tokenizer.tokenize(word)
            for tok in sub_tokens:
                base_label = slot_labels[slot_pointer].replace('I-','PREFIX').replace('B-','PREFIX')
                if first:
                    slot_id = self.lang.slot2id.get(base_label.replace('PREFIX', 'B-'), UNK_TOKEN)
                else:
                    slot_id = self.lang.slot2id.get(base_label.replace('PREFIX', 'I-'), UNK_TOKEN)
                aligned_slots.append(slot_id)
                #bert_tokenize_phrase.append(tok)
            slot_pointer += 1

        aligned_slots.append(self.lang.slot2id['O'])

        #print('-  Bert Tokens  :', bert_tokenize_phrase)
        #print('- Aligned Slots :', str(aligned_slots))

        return aligned_slots

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        return {
            'utterance': torch.tensor(self.utt_ids[idx]),
            'slots': torch.tensor(self.slot_ids[idx]),
            'intent': self.intent_ids[idx],
            'attention_mask': torch.tensor(self.masks[idx]),
            'token_type_ids': torch.tensor(self.token_type_ids[idx])
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
    
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"]) 
    attention_mask, _ = merge(new_item['attention_mask'])
    token_type_ids, _ = merge(new_item['token_type_ids'])
    intent = torch.LongTensor(new_item["intent"])

    src_utt = src_utt.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    intent = intent.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)

    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["attention_mask"] = attention_mask
    new_item['token_type_ids'] = token_type_ids

    #sample = {'utterances': src_utt.shape, 'slots_len': y_lengths.shape, 'intents': intent.shape, 'yslots':y_slots.shape, 'attention_mask':new_item["attention_mask"].shape}
    #print('-   Collate_fn :', sample)
    return new_item
