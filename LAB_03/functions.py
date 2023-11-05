import pandas as pd
import string
import nltk
import math
import numpy as np
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.stem import WordNetLemmatizer
from nltk.lm import Vocabulary
from nltk.lm import NgramCounter
from nltk.lm.preprocessing import flatten
from nltk.lm import Lidstone, MLE, AbsoluteDiscountingInterpolated
from nltk.lm import StupidBackoff
from nltk.lm.api import LanguageModel
from unidecode import unidecode
from word2number import w2n
nltk.download('gutenberg')

def compute_ppl(model, data):
    ngrams, _ = padded_everygram_pipeline(model.order, data)
    scores = [model.logscore(w[-1], w[:-1]) for gen in ngrams for w in gen if len(w) == model.order]
    
    return math.pow(2.0, (-1 * np.asarray(scores).mean()))

class MyStupidBackoff(LanguageModel):
    """
    Implements the Stupid Backoff language model.

    The Stupid Backoff algorithm uses backoff by shortening the n-gram context 
    when the specific n-gram has not been observed. It does not normalize the 
    probabilities, and multiplies by a factor (alpha) when backing off.

    Attributes:
    - alpha (float): backoff discount factor.
    """
    def __init__(self, alpha=0.4, *args, **kwargs):
        """
        Initializes my Stupid Backoff model.

        Parameters:
        - alpha (float, optional): The backoff discount factor. Default is 0.4.
        - *args, **kwargs: eventually additional parameters for super constructor.
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        
    def unmasked_score(self, word, context=None):
        """
        Compute the unnormalized score of a word given a context.

        Parameters:
        - word (str): target word.
        - context (tuple of str, optional): context of the target word.

        Returns:
        - float: The unnormalized probability of the word.
        """
        ngram = (word,) if not context else tuple(context) + (word,)
        return self.backoff(ngram)
        
    def backoff(self, ngram):
        """
        Recursively backs off and computes the probability of the n-gram.

        Parameters:
        - ngram (tuple of str): target n-gram.

        Returns:
        - float: probability of the n-gram
        """
        epsilon = 0.001  # to avoid zero probabilities
        
        if len(ngram) == 1:
            # Base case: Unigram frequency
            total_unigrams = sum(self.counts.unigrams.values())
            return (self.counts[ngram[0]] + epsilon) / total_unigrams
        
        context = ngram[:-1]
        
        # Retrieve the frequency of the ngram and the context
        context_freq = self.counts[len(context)].get(context, 0)
        ngram_freq = self.counts[len(ngram)].get(ngram, 0)
        
        if ngram_freq > 0:
            return ngram_freq / context_freq
        else:
            # Use the Stupid Backoff algorithm to get the probability of the shorter ngram
            return self.alpha * self.backoff(ngram[1:]) + epsilon
        
def prepare_dataset(name):
    """
    Load the specified dataset from NLTK and normalize words.
    
    Parameters:
    - name (str): name of the corpus from the NLTK Gutenberg collection.
    
    Returns:
    - dataset (list of list of str): normalized dataset.
    """
    dataset = [sent for sent in nltk.corpus.gutenberg.sents(name)]
    dataset = [normalize_sent(sent) for sent in dataset]
    return dataset

def split_dataset(dataset, size):
    """
    Split the dataset into training and test sets.
    
    Parameters:
    - dataset (list): dataset to split.
    - size (float): fraction of dataset for training.
    
    Returns:
    - train (list): training set
    - test (list): test set
    """
    train = dataset[:int(size * len(dataset))]
    test = dataset[int(size * len(dataset)):]
    return train, test

def get_vocabulary(dataset, vocab_cutoff):
    """
    Generate vocabulary from the dataset.
    
    Parameters:
    - dataset (list): dataset from which to build the vocabulary.
    - vocab_cutoff (int): the threshold for words to be considered as 'known' in the vocabulary.
    
    Returns:
    - vocabulary (Vocabulary): vocabulary built from the dataset.
    """
    words = list(flatten(dataset))
    vocabulary = Vocabulary(words, unk_cutoff = vocab_cutoff)
    return vocabulary

def get_counter(dataset, ngram_order):
    """
    Generate an n-gram counter for the dataset.
    
    Parameters:
    - dataset (list): dataset from which to build the n-gram counter.
    - ngram_order (int): order of n-grams.
    
    Returns:
    - counter (NgramCounter): Ngram counter object 
    """
    padded_ngram_dataset, _ = padded_everygram_pipeline(ngram_order, dataset)
    counter = NgramCounter(padded_ngram_dataset)
    return counter

def replace_oov(dataset, vocabulary):
    """
    Replace out-of-vocabulary (OOV) words in the dataset with a special token.
    
    Parameters:
    - dataset (list): dataset.
    - vocabulary (Vocabulary): vocabulary to use to check words.
    
    Returns:
    - dataset (list): dataset with OOV words replaced by <UNK>.
    """
    dataset = [list(vocabulary.lookup(sent)) for sent in dataset]
    return dataset

def train_model(model, training_set, ngram_order, alpha = None, discount = None, gamma = None):
    """
    Train a specified model on a given training set.

    Parameters:
    - model (str): specifies the type of model to be trained ('MLE', 'StupidBackoff' or 'MyStupidBackoff').
    - training_set (list): training dataset.
    - ngram_order (int): n-grams max length.
    - alpha (float, optional): smoothing parameter for 'StupidBackoff' and 'MyStupidBackoff'.

    Returns:
    - lm: trained language model.
    """
    print('\n-------------------------------------------')
    ngrams, flat_text = padded_everygram_pipeline(ngram_order, training_set)

    if model == 'MLE':
        print(f'\nTraining MLE model')
        lm = MLE(order = ngram_order)

    elif model == 'ADI':
        print(f'\nTraining Absolute Discounting Interpolated model')
        lm = AbsoluteDiscountingInterpolated(order = ngram_order, discount = discount)

    elif model == 'Lidstone':
        print(f'\nTraining Lidstone model')
        lm = Lidstone(gamma = gamma, order = ngram_order)

    elif model == 'StupidBackoff':
        print(f'\nTraining NLTK Stupid Backoff model')
        lm = StupidBackoff(order = ngram_order, 
                           alpha = alpha)
    
    elif model == 'MyStupidBackoff':
        print(f'\nTraining my Stupid Backoff implementation')
        lm = MyStupidBackoff(
                        order = ngram_order,
                        alpha=alpha)
    
    lm.fit(ngrams, flat_text)
    print(f'{model} Trained!')
    return lm
    
def test_model(model, model_name, dataset, predefined_tests, ngram_order):
    """
    Test a trained model on a given dataset and report metrics.

    Parameters:
    - model: trained language model.
    - dataset (list): test set.
    - predefined_tests (list of dict): Predefined phrases that will be tested as the test set sentences.
    - ngram_order (int): n-grams max length.

    Returns:
    - df (DataFrame): scores for the predefined and test set phrases.
    - ppl (float): perplexity of the model.
    - cross_entropy (float): cross entropy of the model.
    """
    # Perplexity and cross_entropy

    test_set_ngrams, _ = padded_everygram_pipeline(ngram_order, dataset)
    ngrams = list(flatten(test_set_ngrams))
    ppl =  model.perplexity([ngram for ngram in ngrams if len(ngram) == model.order])

    ppl_2 = compute_ppl(model, dataset)

    test_set_ngrams, _ = padded_everygram_pipeline(ngram_order, dataset)
    ngrams = list(flatten(test_set_ngrams))
    cross_entropy = model.entropy([ngram for ngram in ngrams if len(ngram) == model.order])

    print(f'\nInformation:')
    print(f' - SB Vocabs: {model.vocab}')
    print(f' - SB Counts: {model.counts}')
    print(f' - Generating words with SB:\n', model.generate(10))
    
    print(f'\nScores:')
    print(f' - Perplexity: {ppl}')
    print(f' - Perplexity (compute_ppl()): {ppl_2}')
    print(f' - Cross Entropy : {cross_entropy}')
    print(f' - Perplexity (from entropy): {pow(2, cross_entropy)}')
    
    print(f'\nScores on predefined and test phrases:')

    df_prf = pd.DataFrame([{'Phrase': test_el['phrase'], 
                        'Context': test_el['context'],
                        'Score': model.score(test_el['phrase'], test_el['context'])} for test_el in predefined_tests])

    df_ts = pd.DataFrame([{'Phrase': ' '.join(test_sent), 
                        'Context': None,
                        'Score': model.score(' '.join(test_sent))} for test_sent in dataset[:20]])
    tested_sents = pd.concat([df_prf, df_ts])
    print(tested_sents)

    scores = pd.DataFrame(columns = ['Std. Perplexity','C_ppl() Perplexity','Cross Entropy'])
    scores.loc[model_name] = [ppl, ppl_2, cross_entropy]
    return tested_sents, scores

def normalize_sent(sent):
    """
    Performs a series of normalization steps on a list of words. This includes:
    - Removing newline characters
    - Removing punctuation
    - Removing accent marks and other diacritics
    - Converting number words to numbers
    - Lemmatization
    - Stemming
    - Removing empty strings or strings containing only whitespace

    Parameters:
    - words (list): List of words to be normalized.

    Returns:
    - list: List of normalized words.

    """
    # Create a lemmatizer and stemmer instance
    lemmatizer = WordNetLemmatizer()

    # Remove newline characters
    sent = [str(token).replace('\n', ' ') for token in sent]
    
    # Remove punctuation
    sent = [''.join(c for c in str(token) if c not in string.punctuation) for token in sent]
    
    # Remove accent marks and other diacritics
    sent = [unidecode(str(token)) for token in sent]
    
    # Number normalization (number words to numbers)
    sent = [str(w2n.word_to_num(token)) if str(token) in w2n.american_number_system else token for token in sent]
    
    # Lemmatization - Group togheter inflected forms of a word, so it can be analyzed as a single item
    sent = [lemmatizer.lemmatize(str(token)) for token in sent]
        
    # Rmove empty strings or whitespace words
    sent = [token for token in sent if token.strip()]

    return sent
                     
def dataset_info(dataset, dataset_name, ngram_len, vocab_cutoff):

    vocabulary = get_vocabulary(dataset, vocab_cutoff)
    ngram_counter = get_counter(dataset, ngram_len)

    print('\n-------------------------------------------')
    print(f'\n{dataset_name} INFO:')
    print(f' - Padding and ngram done!')
    print(f' - Sentences: {len(dataset)}')
    print(f' - Vocabulary Size: {len(list(vocabulary))}')
    print(f' - Example Sents:\n -> {dataset[:5]}')
    print(f'Ngrams information (Counter):')
    print(f' - N gram stored: {ngram_counter.N()}')
    print(f' - Unigrams count: {ngram_counter.unigrams}')
    print(f' - Bigram example:', ngram_counter[['done']]['when'])
    for word in dataset[1]:
        # membership test
        word_membership = word in vocabulary
        # accessing count
        word_count = vocabulary[word]
        # lookup
        word_lookup = vocabulary.lookup(word)
        
        print(" - Word: '{}', Count: {}, In Vocab: {}, Mapped to: '{}'".format(
            word, word_count, word_membership, word_lookup))
