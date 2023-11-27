# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import nltk
import pandas as pd
import spacy
from tqdm import tqdm
from conll import evaluate
from tabulate import tabulate
from sklearn_crfsuite import CRF
from nltk.corpus import conll2002
from spacy.tokenizer import Tokenizer
nlp = spacy.load("es_core_news_sm")
import es_core_news_sm


def dataset():
    """
    Load and split in training and test set the Conll2002 dataset (Spanish version).
    
    Returns:
        tuple: The training and test datasets.
    """    
    
    nltk.download('conll2002')
    training_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
    test_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testb')]
    
    return training_sents, test_sents

def process_data(set, configuration, name):
    """
    Process a dataset by extracting features and labels.
    
    Args:
        set (list): dataset to be processed
        configuration (str): label for the predefined feature configuration
        name (str): Name of the dataset (for logging purposes)
    
    Returns:
        dict: dictionary containing dataset with features setted, the labels and the "raw" dataset sentences.
    """
    print(f'Setting features and labels for {name} ...')
    extractor = FeatureExtractor()

    features = extractor.extract_features(set, configuration)

    # set2label (sent2label)
    label = [[label for _, label in sent] for sent in set]
    
    return {
        'Features': features,
        'Labels': label,
        'Sents': set
    }
    
def feature_selection(train_set, test_set, features_conf):
    """
    Extract and prepare features and labels for the training and test datasets.
    Print info abount this phase.
    
    Args:
        train_set (list): dataset of sentences
        test_set (list): dataset of sentences
        features_conf (dict): configuration for feature extraction
    
    Returns:
        tuple: processed training and test datasets.
    """
    print(f'Feature Selection :\n - Configuration: {features_conf}\n')

    training_set = process_data(train_set, features_conf, 'Training set')
    test_set = process_data(test_set, features_conf, 'Test set')

    print(f'\nFeature selection phase info:')
    print(f' - Training set featured size: {len(training_set["Features"])}')
    print(f' - Training set labels size: {len(training_set["Labels"])}')
    print(f' - Test set featured size: {len(test_set["Features"])}')
    print(f' - Test set labels size: {len(test_set["Labels"])}')
    print(f' - First train token features: {training_set["Features"][0][0]}')
    print(f' - First train token label: {training_set["Labels"][0][0]}')
    print(f' - First test token features: {test_set["Features"][0][0]}')
    print(f' - First test token label: {test_set["Labels"][0][0]}')

    return training_set, test_set
    
def train_crf(train_set):
    """
    Train a Conditional Random Field (CRF) model
    
    Args:
        train_set (dict): training dataset
    
    Returns:
        CRF: trained CRF model.
    """

    print('\nTraining CRF:')

    crf = CRF(
        algorithm='lbfgs', 
        c1=0.1, 
        c2=0.1, 
        max_iterations=100, 
        all_possible_transitions=True)

    try:
        crf.fit(train_set['Features'], train_set['Labels'])
    except AttributeError:
        pass

    print('--> Model trained!')
    return crf

def test_crf(crf, test_set):
    """
    Test a trained CRF model on the provided dataset and evaluate its performance.
    Print scores.

    Args:
        crf (CRF): trained CRF model
        test_set (dict): test dataset
    
    Returns:
        DataFrame: evaluation scores in a DataFrame format.
    """
    pred = crf.predict(test_set['Features'])
    hyp = [[(test_set['Features'][i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]
    results = evaluate(test_set['Sents'], hyp)

    print('--> Scores:\n')
    pd_tbl = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
    print(tabulate(pd_tbl.round(3), headers='keys', tablefmt='grid'))
    print('\n',''.join(['-' for i in range(100)]),'\n')
    return pd_tbl

class FeatureExtractor:
    def __init__(self):
        # Loading Spanish model from Spacy and initializing tokenizer using whitespaces
        self.nlp = es_core_news_sm.load()
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab)

    def sent2tokens(self, sent):
        # Extract tokens from a given sentence
        return [token for token, _ in sent]

    def sent2features(self, sent, configuration):
        # Convert an input sentence into a list 
        # of feature dictionaries based on the provided configuration
        spacy_sent = self.nlp(' '.join(self.sent2tokens(sent)))
        sent_features = []

        for token in spacy_sent:
            if configuration == "Spacy_1":
                features = self.get_spacy_1_features(token)
            elif configuration == "Spacy_2":
                features = self.get_spacy_2_features(token)
            elif configuration == "ConllTutorial":
                features = self.get_conll_features(token)
            elif configuration == "FeatureWindow_1":
                features = self.get_window_features(token, spacy_sent, 1)
            elif configuration == "FeatureWindow_2":
                features = self.get_window_features(token, spacy_sent, 2)
            else:
                raise ValueError(f"Invalid configuration: {configuration}")

            sent_features.append(features)

        return sent_features

    def get_spacy_1_features(self, token):
        # Extracting the features specified in sent2spacy_features
        return {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_
        }

    def get_spacy_2_features(self, token):
        # Extracting features with as get_spacy_1_features() + suffix
         return {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_,
            'suffix': token.suffix_
        }
    
    def get_conll_features(self, token):
        # Extracting features used in the tutorial on CoNLL dataset
        return {
            'bias': 1.0,
            'word.lower': token.lower_,
            'word[-3:]': token.text[-3:],
            'word[-2:]': token.text[-2:],
            'word.isupper': token.is_upper,
            'word.istitle': token.is_title,
            'word.isdigit': token.is_digit,
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2]
        }

    def get_window_features(self, token, spacy_sent, window_size):
        # Extracting features accordingly to requested size of feature windows       
        features = self.get_conll_features(token)
        for i in range(1, window_size + 1):
            if token.i - i >= 0:
                features.update(self.get_relative_features(spacy_sent[token.i - i], f"-{i}"))
            if token.i + i < len(spacy_sent):
                features.update(self.get_relative_features(spacy_sent[token.i + i], f"+{i}"))

        if token.i == 0:
            features['BOS'] = True
        elif token.i == len(spacy_sent) - 1:
            features['EOS'] = True

        return features

    def get_relative_features(self, token, prefix):
        # Extracting features for tokens relative to the input token.
        # The prefix indicate the input token position with respect to the main one.
        return {
            f'{prefix}:word.lower': token.lower_,
            f'{prefix}:word.istitle': token.is_title,
            f'{prefix}:word.isupper': token.is_upper,
            f'{prefix}:postag': token.pos_,
            f'{prefix}:postag[:2]': token.pos_[:2]
        }

    def extract_features(self, dataset, configuration):
        # Extract features for aa dataset based on the given configuration
        
        # Here is managed the baseline configuration
        if configuration == "Baseline":
            return [[{'bias': 1.0, 'word.lower()': word[0].lower()} for word in sent] for sent in tqdm(dataset)]

        # Here all configurations requested in lab exercise
        return [self.sent2features(sent, configuration) for sent in tqdm(dataset)]



# CONSOLE INFO

def dataset_info(training_sents, test_sents):
    # Print info about the dataset

    total_esp = len(conll2002.iob_sents('esp.train')) + len(conll2002.iob_sents('esp.testa')) + len(conll2002.iob_sents('esp.testb'))
    total_ned = len(conll2002.iob_sents('ned.train')) + len(conll2002.iob_sents('ned.testa')) + len(conll2002.iob_sents('ned.testb'))
    
    print('\n',''.join(['-' for i in range(100)]),'\n')
    print(f'Dataset info \n')
    print(f' - Size (ESP): {total_esp}')
    print(f' - Size (NED): {total_ned}')
    print(f' - Train size (ESP): {len(training_sents)}')
    print(f' - Test size (ESP): {len(test_sents)}')
    print(f' - Chunk type: {conll2002._chunk_types}')
    print(f' - Training set first sent:\n--->', conll2002.sents('esp.train')[0])
    print(f' - Training set first sent tagged:\n--->', conll2002.tagged_sents('esp.train')[0])
    print(f' - Training set first sent IOB:\n--->',training_sents[0])
    print('\n',''.join(['-' for i in range(100)]),'\n')