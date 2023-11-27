
from functions import *


def preprocess(dataset, label, mark_neg = True):
    new_dataset = []

    def chunk_sequence(sequence):
        return [sequence[i:i + BERT_MAX_LEN] for i in range(0, len(sequence), BERT_MAX_LEN)]
    
    maxlen = 0
    for tokens in tqdm(dataset[:int(len(dataset)/40) if TESTING else len(dataset)], desc = 'Preprocessing dataset'):
        text = ' '.join(tokens)

        """
        vscore = sia.polarity_scores(text)['compound']
        if vscore <= -0.5:
            vscore = 'VNEG'  # Molto negativo
        elif vscore <= 0:
            vscore = 'NEG'  # Negativo
        elif vscore <= 0.5 :
            vscore = 'POS'  # Positivo
        else:
            vscore = 'VPOS'  # Molto positivo
        """
        
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [str(w2n.word_to_num(token)) if token in w2n.american_number_system else token for token in tokens]

        if mark_neg:
            tokens = mark_negation(tokens)

        if len(tokens) != 0:
            if len(tokens) > maxlen:
                maxlen = len(tokens)
            tokenized_sent = TOKENIZER(' '.join(tokens), truncation=False, padding=False)
            encoded_sentence = tokenized_sent['input_ids']
            attention_mask = tokenized_sent['attention_mask']

            if len(encoded_sentence) > BERT_MAX_LEN:
                chunked_sentences = chunk_sequence(encoded_sentence)
                chunked_attention_masks = chunk_sequence(attention_mask)

                for sent, mask in zip(chunked_sentences, chunked_attention_masks):
                    new_dataset.append((sent, mask, label))
            else:
                new_dataset.append((encoded_sentence, attention_mask, label))
        
    return new_dataset

def filter_movie_reviews(filter):
    mr = movie_reviews
    new_mr = {}
    categories = mr.categories()

    print(' - Filter size:', len(filter))
    print(' - Movie review pos sent:', len(movie_reviews.sents(categories = 'pos')))
    print(' - Movie review neg sent:', len(movie_reviews.sents(categories = 'neg')))

    for category in categories: # ['neg','pos']

            processed_set = preprocess(movie_reviews.sents(categories = category), label=category, mark_neg=True)                
            new_mr[category] = []
            for sample in processed_set:
                sent = sample[0]
                if sent not in filter:
                    new_mr[category].append(sample)
            
    print(' - Filtered Movie review pos sent:', len(new_mr['pos']))
    print(' - Filtered Movie review neg sent:', len(new_mr['neg']))
    return new_mr
    
def load_dataset(dataset_name, kfold, test_size = 0.1, args = [], tr_batch = 64, vl_batch = 32):
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
        grp1_sentences = preprocess([sent for sent in mr.sents(categories='neg')], label = 'neg')
        grp2_sentences = preprocess([sent for sent in mr.sents(categories='pos')], label = 'pos')

    # Mr is a list od doc (list of list of list). Here is transformed into list of sentences, to feed the subjectivity model.
    # These sentences will be then filtered.
    elif dataset_name == 'movie_review_4subjectivity':

        mr = movie_reviews
        categories = mr.categories()

        # We only need the sentences, no other information and no distinction.
        all_sentences = preprocess(movie_reviews.sents(), 'neg')

        lang = Lang(categories)
        dataset = Dataset(all_sentences, lang)
        dataloader = DataLoader(dataset, batch_size = tr_batch, collate_fn = collate_fn)

        return dataloader, None, lang
    
    elif dataset_name == 'movie_review_filtered':

        mr_filtered = filter_movie_reviews(args[0]) # args[0] contains the sentences to remove (is the filter).
        categories = movie_reviews.categories()

        # Create standard dataset
        grp1_sentences = mr_filtered['pos']
        grp2_sentences = mr_filtered['neg']

    else:
        raise Exception('Dataset name not recognized.')
    
    all_sentences = grp1_sentences + grp2_sentences
    train_sentences, test_sentences = train_test_split(all_sentences, test_size=test_size, random_state=42, shuffle = True)

    lang = Lang(categories)

    train_labels = [label for _, _, label in train_sentences]
    fold_datasets = []
    
    # Creating folds
    for k, (train_indices, val_indices) in enumerate(kfold.split(train_sentences, train_labels)):

        train_samples = [train_sentences[i] for i in train_indices]
        val_samples = [train_sentences[i] for i in val_indices]

        print(f' - FOLD {k} - Train Size: {len(train_samples)} - Val Size: {len(val_samples)}')

        train_dataset = Dataset(train_samples, lang)
        val_dataset = Dataset(val_samples, lang)
        train_loader = DataLoader(train_dataset, batch_size = tr_batch, shuffle = True, collate_fn = collate_fn)
        val_loader = DataLoader(val_dataset, batch_size = vl_batch, shuffle = True, collate_fn = collate_fn)
        fold_datasets.append((train_loader, val_loader))

    test_dataset = Dataset(test_sentences, lang)
    test_loader = DataLoader(test_dataset, batch_size = tr_batch, shuffle = True, collate_fn = collate_fn)

    # Info
    print(f' - TEST SET - Size: {len(test_sentences)}')
    print(' - Classes label ids:',lang.class2id)
    print(' - Vocabulary size:', lang.vocab_size)
    print(' - Group ',grp1_sentences[0][2],' - First sent len:', len(grp1_sentences[0][0]), )
    print(' - Group ',grp2_sentences[0][2],' - First sent len:', len(grp2_sentences[0][0]), )
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
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        
        self.class2id = {}
        for i, cls in enumerate(classes):
            self.class2id[cls] = i

        self.id2class = {i:c for c, i in self.class2id.items()}

    def encode(self, sentence):
        return self.tokenizer.encode(sentence)

    def decode(self, sentence_ids):
        return self.tokenizer.decode(sentence_ids)

class Dataset(data.Dataset):
    def __init__(self, samples, lang):
        self.samples = samples
        self.lang = lang
        self.first = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence, mask, label = self.samples[idx]

        attention_mask = torch.tensor(mask)
        tensor_sentence = torch.tensor(sentence)
        encoded_label = self.lang.class2id[label]

        if self.first and INFO_ENABLED:
            print('- Sample (Label:', label, ')')
            print('-- Sentence:', sentence)
            print('-- Label:', label)
            print('-- Encoded:', encoded_label)
            print('--attMask:', attention_mask)
            self.first = False


        return {'text':tensor_sentence, 'attention_mask':attention_mask, 'label':encoded_label}
    
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

        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq

        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    new_item = {}
    for key in batch[0].keys():
        new_item[key] = [el[key] for el in batch]

    source, _ = merge(new_item['text'])
    attention_masks, _ = merge(new_item['attention_mask'])

    new_item['text'] = source.to(DEVICE)
    new_item['labels'] = torch.tensor(new_item['label']).to(DEVICE)
    new_item['attention_masks'] = attention_masks.to(DEVICE)

    if INFO_ENABLED:
        print('COLLATEFN:',new_item['text'].shape.shape, new_item['labels'].shape, new_item['lengths'].shape) # , new_item['vlabels']
    return new_item

