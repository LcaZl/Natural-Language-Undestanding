import nltk
import pandas as pd

from spacy.tokenizer import Tokenizer
from tqdm import tqdm
import en_core_web_sm
from nltk.metrics import accuracy
from nltk import NgramTagger
from nltk.corpus import treebank

nltk.download('universal_tagset')
nltk.download('treebank')


def prepare_dataset(train_size):
    
    tagged_sentences = treebank.tagged_sents(tagset='universal') # [[('The', 'AT'), ('Fulton', 'NP-TL')...],
    train_set_size = int(len(tagged_sentences) * train_size)
    train_set = tagged_sentences[:train_set_size]
    test_set = tagged_sentences[train_set_size:]

    print(f'Dataset downloaded:')
    print(f' - Dataset size: {len(tagged_sentences)}')
    print(f' - Training set size: {len(train_set)}')
    print(f' - Test set size: {len(test_set)}')
    print(f' - Training set first tagged words:', train_set[0][:3])
    print(f' - Test set first tagged words:', test_set[0][:3])

    return train_set, test_set

def evaluate_nltk_taggers(training_set, test_set, Nmax, backoffs, cutoffs):
    """
    Evaluate different configurations of NLTK Ngram taggers.
    
    Parameters:
    - training_set: training set
    - test_set: test set
    - Nmax: maximum n-gram size to test
    - backoffs: List of maximum two boolean that indicate if the backoff should be setted or not
    - cutoffs: list of integers indicating the frequency cut-offs for the tagger
    
    Returns:
    - A DataFrame with columns ['N', 'Backoff', 'Cut-off', 'Accuracy'] showing the accuracy
      of the corresponding tagger configuration
    """
    df = pd.DataFrame(columns=['N','Backoff','Cut-off','Accuracy'])

    iters =  len(cutoffs) * len(backoffs) * Nmax
    progress_bar = tqdm(total=iters, desc="Processing", position=0)

    for N in range(Nmax):

        for backoff in backoffs:

            for cut_off in cutoffs:

                progress_bar.set_description(f'N:{N} - Backoff:{backoff} - Cutoff:{cut_off}')
                
                tagger = train_ngram_tagger(training_set, N, backoff, cut_off)
                
                accuracy = tagger.evaluate(test_set)

                df.loc[len(df)] = [N, backoff, cut_off, accuracy]
                
                progress_bar.update(1)
    
    progress_bar.close()
    print(f' -> Evaluation completed.\n')
    return df

def train_ngram_tagger(train_set, n, backoff, cutoff):
    """
    Train NLTK NgramTagger with the given parameters.
    
    Parameters:
    - train_set: training set
    - n: n-gram max size.
    - backoff: Boolean indicating whether to use backoff
    - cutoff: frequency cut-off for the tagger.
    
    Returns:
    - trained NgramTagger.
    """
    if n == 0:
        tagger = nltk.DefaultTagger('NOUN')
    else:
        if backoff:
            t_tmp = nltk.DefaultTagger('NOUN')
            for i in range(1, n):
                tagger = NgramTagger(n=i, 
                                 train=train_set, 
                                 backoff = t_tmp,
                                 cutoff= cutoff)
                t_tmp = tagger
            return t_tmp
        else:
            tagger = NgramTagger(n=n, 
                                 train=train_set, 
                                 backoff = None,
                                 cutoff= cutoff)

    return tagger

def get_spacy_tagger(test_set, spacy_to_nltk_map):
    """
    Get spacy tags for the input dataset
    
    Parameters:
    - test_set: test dataset
    - spacy_to_nltk_map: dictionary to map Spacy POS tags to NLTK tags.
    
    Returns:
    - list of tuples with spacy tag mapped to nltk one
    """    
    nlp = get_custom_nlp()
    if nlp is None:
        print(f'Custom NLP error. Exiting.')
        return None
    
    spacy_tags = []

    for sent in test_set:
        doc = nlp(' '.join([word for word, _ in sent]))
        spacy_tags.extend([(token.text, spacy_to_nltk_map[token.pos_]) for token in doc])
    
    return spacy_tags

def evaluate_spacy_tagger(tags, test_set):
    """
    Evaluate the accuracy of the Spacy tagger.
    
    Parameters:
    - tags: List of tuples with the spacy tag (mapped to NLTK)
    - test_set: test dataset with the NLTK correct tags
    
    Returns:
    - Accuracy of the spacy tagger on test set
    """
    correct_tags = [tag for sent in test_set for word, tag in sent]
    predicted_tags = [tag for _, tag in tags]
    acc = accuracy(correct_tags, predicted_tags)
    return acc

def get_custom_nlp():
    """
    Load Spacy nlp() with a custom tokenizer.
    
    Returns:
    - nlp object with a custom tokenizer, if the tokenizer passes the sanity check.
    """

    # Sanity check
    print(f'\nSanity check:')
    success = True

    nlp = en_core_web_sm.load()
    nlp.tokenizer = Tokenizer(nlp.vocab) # Tokenize by whitespace

    for id_sent, sent in enumerate(tqdm(treebank.sents())):
        doc = nlp(" ".join(sent))
        if len([x.text for x in doc]) != len(sent):
            print(id_sent, sent)
            success = False

    if success:
        print(f' -> Sanity check ok.\n')
        print(f'Spacy pipeline:')
        print(' -> '.join([f'{key}' for key, _ in nlp.pipeline]))

    else:
        print(f' -> Sanity check not passed. Exiting.')
        return None
    
    return nlp