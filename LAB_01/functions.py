# Libraries
import spacy

import string
import nltk
import pandas as pd
from collections import Counter
from tabulate import tabulate

# Stop words libraries
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS
from nltk.corpus import stopwords
NLTK_STOP_WORDS = set(stopwords.words('english'))

# For stemming and lemmatization
from nltk.stem import WordNetLemmatizer
from word2number import w2n

# To convert numbers to words and viceversa
from num2words import num2words
from unidecode import unidecode

# Get NLTK resources
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('wordnet')

# Load Spacy model
spacy.load('en_core_web_sm')
import en_core_web_sm
nlp = en_core_web_sm.load()

def compute_descriptive_stats(book, library, stop_word_set, frequency_list_method, nbest_v, cut_off_bounds, show_progress = False):
    """
    Computes descriptive statistics for a given book corpus based on specified parameters.
    
    Parameters:
    - book (str): The name of the book.
    - library (str): The library/tool to be used for corpus extraction (e.g., NLTK, spaCy).
    - stop_word_set (list/set): A collection of stop words to be removed from the corpus.
    - frequency_list_method (str): Method to generate frequency list.
    - nbest_v (int): Number of top terms to consider in frequency list.
    - cut_off_bounds (tuple): Lower and upper bounds for frequency list cut-off.
    - show_progress (bool, optional): If set to True, displays the results. Defaults to False.
    
    Returns:
    - lexicon (list): A list containing unique words/terms from the book.
    - stats (Dataframe): Dataframe containing various corpus statistics.
    - frequency_list_df (DataFrame): DataFrame containing top-n frequent terms based on provided parameters.
    """
    
    # Index of the current output
    index = [f'<LIB:{library}, SWS:{stop_word_set}, FLM:{frequency_list_method}>']

    # Load corpus at different level with the specified library (the raw version is always loaded with NLTK)
    chars, words, sents = get_corpus_information(book, library)
    
    # Remove stop words accordingly to the specified set
    words = remove_stop_words(words, stop_word_set)

    # Normalize words in the corpus by:
    # - Normalizing numbers
    # - Removing newline characters, punctuation, and accents/diacritics
    # - Removing empty or whitespace-only words
    words = normalize_words(words)
    
    stats, lexicon = compute_corpus_statistics_and_lexicon(index, chars, words, sents)
    
    # Get frequency list
    frequency_list = get_frequency_list([str(token) for token in words], frequency_list_method)
    frequency_list_cutted = cut_off(frequency_list, cut_off_bounds)
    frequency_list_nbest = dict(sorted(frequency_list_cutted.items(), key = lambda item: item[1], reverse= True)[:nbest_v])
    
    # Transforming the frequency list into a dataframe
    fr_data = [[f'<{str(el)}>:{str(freq)}' for el, freq in frequency_list_nbest.items()]]
    column_names = [f'{i+1}' for i in range(nbest_v)]
    frequency_list_df = pd.DataFrame(data=fr_data, index=index, columns=column_names)
    
    # Optionally, based on show_progress parameter, show the resulting dataframes
    if show_progress:
        print(f'------> Configuration index: {index}')
        print_results(stats, '------> Corpus statistics')
        print_results(frequency_list_df, '------> Top-N Frequencies')    

    return lexicon, stats, frequency_list_df

def get_corpus_information(book, library):
    """
    Extracts and tokenizes the content of a book using the specified library.
    
    Parameters:
    - book (str): The identifier of the book to be processed.
    - library (str): The library/tool to be used for text tokenization. 
    
    Returns:
    - chars (str): The raw content of the book.
    - words (list): List of tokenized words from the book.
    - sents (list): List of tokenized sentences from the book.
    """

    chars = nltk.corpus.gutenberg.raw(book)

    if library == 'Ref': # References
        words = nltk.corpus.gutenberg.words(book) # words
        sents = nltk.corpus.gutenberg.sents(book) # sentences
        
    elif library == 'NLTK': #NLTK
        words = nltk.word_tokenize(chars)
        sents = nltk.sent_tokenize(chars)
        
    elif library == 'Spacy': # Spacy
        words = nlp(chars, disable=["ner"])
        sents = list(words.sents)
    
    return chars, words, sents

  
def compute_corpus_statistics_and_lexicon(index, chars, words, sents):
    """
    Computes various statistics related to a corpus and generates its lexicon.
    
    Parameters:
    - index (list): A list containing a unique index for the current stats.
    - chars (str): The raw content of the book or corpus.
    - words (list): List of tokenized words from the corpus.
    - sents (list): List of tokenized sentences from the corpus.
    
    Returns:
    - stats (DataFrame): A pandas DataFrame containing various computed statistics.
    - lexicon (set): Set containing unique lowercase words from the corpus.
    
    The generated statistics include:
    - Character count, word count, and sentence count.
    - Average words per sentence.
    - Average characters per word and per sentence.
    - Length of the longest and shortest sentence and token.
    - Size of the lexicon (unique lowercase words).
    """
    stats = pd.DataFrame(index = index)

    stats['char_count'] = len(''.join(chars))
    stats['words_count'] = len(words)
    stats['sents_count'] = len(sents)
        
    stats['avg_word_x_sent'] = round(sum([len(sent) for sent in sents]) / stats['sents_count'])
    stats['avg_char_x_word'] = round(sum([len(token) for token in words]) / stats['words_count'])
    stats['avg_char_x_sent'] = round(sum([len(''.join(str(sent))) for sent in sents]) / stats['sents_count'])
    
    stats['longest_sent'] = max([len(sent) for sent in sents])
    stats['shortest_sent'] = min([len(sent) for sent in sents])
    stats['longest_token'] = max([len(token) for token in words])
    stats['shortest_token'] = min([len(token) for token in words])
    
    lexicon = set([str(word).lower() for word in words])
    stats['lowered_lexicon_size'] = len(lexicon)
    
    return stats, lexicon

def remove_stop_words(words, stop_words_set):
    """
    Filter stop words from words in input.
    
    Parameters:
    - words (list): List of words from which stop words are to be removed.
    - stop_words_set (str): Specifies the set of stop words to be used. 
    
    Returns:
    - list: List of words after removing stop words.
    
    """
    if stop_words_set == 'NLTK': #NLTK no words

        stop_words = set(NLTK_STOP_WORDS)
        
    elif stop_words_set == 'Spacy': # Spacy no words
        stop_words = set(SPACY_STOP_WORDS)
        
    elif stop_words_set == 'Scikit':# Scikit-learn
        stop_words = set(SKLEARN_STOP_WORDS)
        
    else:
        stop_words = set()

    # Return input words without the words in stop_words set.
    return [word for word in words if word not in stop_words]

def normalize_words(words):
    """
    Performs a series of normalization steps on a list of words. This includes:
    - Removing newline characters
    - Removing punctuation
    - Removing accent marks and other diacritics
    - Converting number words to numbers
    - Lemmatization
    - Removing empty strings or strings containing only whitespace

    Parameters:
    - words (list): List of words to be normalized.

    Returns:
    - list: List of normalized words.

    """
    # Create a lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove newline characters
    words = [str(token).replace('\n', ' ') for token in words]
    
    # Lower case
    words = [str(token.lower()) for token in words]

    # Remove punctuation
    words = [''.join(c for c in str(token) if c not in string.punctuation) for token in words]
    
    # Remove accent marks and other diacritics
    words = [unidecode(str(token)) for token in words]

    # Number normalization (number words to numbers)
    words = [str(w2n.word_to_num(token)) if str(token) in w2n.american_number_system else token for token in words]
    
    # Lemmatization - Group togheter inflected forms of a word, so it can be analyzed as a single item
    words = [lemmatizer.lemmatize(str(token)) for token in words]
    
    # Rmove empty strings or whitespace words
    words = [token for token in words if token.strip()]
    
    return words

def get_frequency_list(lexicon, method):
    """
    Computes the frequency distribution of words in the lexicon based on the specified method.
    
    Parameters:
    - lexicon (list): List of words to compute frequencies.
    - method (str): Method to compute word frequencies. 

    Returns:
    - dict/list: A frequency distribution of words.
    
    """
    if method == 'Counter':
        frequency_list = Counter(lexicon)
        
    elif method == 'NLTK':
        frequency_list = nltk.FreqDist(lexicon)
        
    else:
        frequency_list = [] 
        
    return frequency_list    

def cut_off(frequency_list, bounds):
    """
    Filters the words in the frequency list based on a specified range.
    
    Parameters:
    - frequency_list (dict): Dictionary with word frequencies.
    - bounds (tuple): A tuple of two integers representing the minimum and maximum 
        frequency (inclusive) for words to be retained.

    Returns:
    - dict: Dictionary containing words within the specified frequency bounds.
    """
    new_vocab = {}
    for word, count in sorted(frequency_list.items(), reverse = True):
        if count >= bounds[0] and count <= bounds[1]:
            new_vocab.update({word:count})
    return new_vocab

def print_results(df, title = None):
    """
    Prints the provided dataframe in a tabulated format.
    
    Parameters:
    - stats (DataFrame): Dataframe
    - title (str, optional): Title or header to be displayed above the statistics. 

    """
    
    if title is not None:
        print(f'{title}')
    print(tabulate(df, headers = "keys",tablefmt='simple_grid', floatfmt=".1f"))



