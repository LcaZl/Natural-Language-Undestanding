# Add functions or classes used for data loading and preprocessing
import string
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.util import mark_negation
from sklearn.model_selection import StratifiedKFold

nltk.download('sentiwordnet')
# Parameters
PAD_TOKEN = 0
UNK_TOKEN = 1
DEVICE = 'cuda:0'
INFO_ENABLED = False
MAX_VOCAB_SIZE = 10000
TRAIN_PATH = 'dataset/laptop14_train.txt'
TEST_PATH = 'dataset/laptop14_test.txt'
pos2wn = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

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
            score = score_swn([words])

            assert len(words) == len(tags)

            new_dataset.append({'words':words, 'tags':tags, 'score':score})

    return new_dataset

def load_dataset():
    print(f'Loading Dataset Laptop 14...')

    train_raw = read_file(TRAIN_PATH)
    test_raw = read_file(TEST_PATH)

    train_set = process_raw_data(train_raw)
    test_set = process_raw_data(test_raw)
    print(' - Training sents:', len(train_raw))
    print(' - Test sents:', len(test_raw))
    print(' - Raw sent:', train_raw[0])
    print(f' - Training len:', len(train_set))
    print(f' - Train sample:', train_set[0])
    print(f' - Test len:', len(test_set))
    print(f' - Test sample:', test_set[1])

    sents = [el['words'] for el in train_set]
    lang = Lang(sents)

    print(' - Sent words:', train_set[0]['words'])
    print(' - Sent encod:', lang.encode(train_set[0]['words']))
    print(' - Sent words:', lang.decode(lang.encode(train_set[0]['words'])))

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle = True)

    def integrate_scores_with_dataset(dataset):
        for sample in dataset:
            sample['aspect_score'] = aggregate_aspect_scores(sample)
    
    integrate_scores_with_dataset(train_set)
    integrate_scores_with_dataset(test_set)

    print(' - Training sents:', len(train_raw))
    print(' - Test sents:', len(test_raw))
    print(' - Raw sent:', train_raw[0])
    print(f' - Training len:', len(train_set))
    print(f' - Train sample:', train_set[0])
    print(f' - Test len:', len(test_set))
    print(f' - Test sample:', test_set[1])
    
class Lang:
    def __init__(self, sents):
        
        self.word2id = self.mapping_seq(sents, special_token = True)
        self.id2word = {id: word for word, id in self.word2id.items()}
        self.vocab_size = len(self.word2id)

    def encode(self, sentence):
        return [self.word2id.get(word, UNK_TOKEN) for word in sentence]

    def decode(self, sentence_ids):
        return [self.id2word[id] for id in sentence_ids]
    
    def mapping_seq(self, sents, special_token = False):
        vocab = {}

        if special_token:
            vocab['<PAD>'] = PAD_TOKEN
            vocab['<UNK>'] = UNK_TOKEN


        for sent in sents:
            for word in sent: 
                if not vocab.get(word):
                    vocab[word] = len(vocab)

        return vocab
    
# lesk's filtering w.r.t. pos-tag makes distinction between 'a' & 's'; we do not
# thus, let's modify the function
def lesk(context_sentence, ambiguous_word, pos=None, synsets=None):
    """Return a synset for an ambiguous word in a context.

    :param iter context_sentence: The context sentence where the ambiguous word
         occurs, passed as an iterable of words.
    :param str ambiguous_word: The ambiguous word that requires WSD.
    :param str pos: A specified Part-of-Speech (POS).
    :param iter synsets: Possible synsets of the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.
    """

    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        if pos == 'a':
            synsets = [ss for ss in synsets if str(ss.pos()) in ['a', 's']]
        else:
            synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    _, sense = max(
        (len(context.intersection(ss.definition().split())), ss) for ss in synsets
    )

    return sense

def aggregate_aspect_scores(sample):
    # Aggrega gli score per ogni aspetto
    aspect_scores = {'pos': [], 'neg': [], 'obj': []}
    for word, tag, pos, neg, obj in zip(sample['words'], sample['tags'], sample['score']['pos'], sample['score']['neg'], sample['score']['obj']):
        if 'T-' in tag:  # Se la parola è un aspetto
            aspect_scores['pos'].append(pos)
            aspect_scores['neg'].append(neg)
            aspect_scores['obj'].append(obj)
    
    # Calcola la media degli score per aspetto
    for key in aspect_scores:
        if aspect_scores[key]:  # Evita la divisione per zero
            aspect_scores[key] = sum(aspect_scores[key]) / len(aspect_scores[key])
        else:
            aspect_scores[key] = 0
    
    return aspect_scores

def score_sent(sent, use_pos=False):
    pos = []
    neg = []
    obj = []
    if use_pos:
        tagged_sent = pos_tag(sent, tagset='universal')
    else:
        tagged_sent = [(w, None) for w in sent]

    for tok, tag in tagged_sent:
        ss = lesk(sent, tok, pos=pos2wn.get(tag, None))
        if ss:
            sense = swn.senti_synset(ss.name())
            pos.append(sense.pos_score())
            neg.append(sense.neg_score())
            obj.append(sense.obj_score())
        else:
            pos.append(0)
            neg.append(0)
            obj.append(1)
    return pos, neg, obj

def score_swn(doc, use_pos=False):
    pos = []
    neg = []
    obj = []
    for sent in doc:
        sent_pos, sent_neg, sent_obj = score_sent(sent, use_pos=use_pos)
        #print(sent_pos, sent_neg, sent_obj)
        pos.extend(sent_pos)
        neg.extend(sent_neg)
        obj.extend(sent_obj)

    scores = {
        "pos": pos,
        "neg": neg,
        "obj": obj
    }
    return scores
