# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from wsd_algorithms import *

import pandas as pd
import numpy as np
import warnings
from tabulate import tabulate

from nltk.metrics.scores import precision, recall, f_measure, accuracy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix

from nltk.corpus import senseval
from nltk.corpus.reader.senseval import *

def get_labels(dataset):
    """
    Encodes the first sense label of each instance in the dataset into integer format.
    
    Parameters:
    - dataset: list of instances (context, word, position, sense)

    Returns:
    - labels: array of shape (n_samples,) containing int trasformed sense labels
    """
    lbls = [inst.senses[0] for inst in dataset]

    lblencoder = LabelEncoder()
    lblencoder.fit(lbls)
    labels = lblencoder.transform(lbls)
    return labels

def bow_features(dataset):
    """
    Extracts BOW features from the context of each sentence in the dataset.
    
    Parameters:
    - dataset: list of instances (context, word, position, sense)

    Returns:
    - sents: matrix of shape (n_samples, n_features) with BOW features
    """
    data = [" ".join([t[0] for t in inst.context]) for inst in dataset]

    vectorizer = CountVectorizer()
    sents = vectorizer.fit_transform(data)
    return sents

def lab_collocational_features(dataset):
    """
    Features used in lab.
    """
    feats = []
    for inst in dataset:
        p = inst.position
        feats.append({
            "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
            "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
            "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
            "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0]
        })
    dvectorizer = DictVectorizer(sparse=False)
    labfeats = dvectorizer.fit_transform(feats)
    return labfeats

def pos_tag_features(dataset, window_size):
    """
    Extracts POS tag features for each element of the dataset

    Parameters:
    - dataset: list of instances (context, word, position, sense)
    - window_size: int defining how many adjacent words to consider

    Returns:
    - postags: array of shape (n_samples, n_features) with POS tagging features
    """
    pt = []
    for inst in dataset:
        p = inst.position
        features = {}
        
        for i in range(1, window_size + 1):
            features[f"w-{i}_pos"] = 'NULL' if p < i else inst.context[p-i][1]
            features[f"w+{i}_pos"] = 'NULL' if len(inst.context) - 1 < p+i else inst.context[p+i][1]
        pt.append(features)

    dvectorizer = DictVectorizer(sparse=False)
    postags = dvectorizer.fit_transform(pt)
    return postags

def ngram_features(dataset, window_size, n):
    """
    Extracts n-gram features from the context around the target word in each sentence of the dataset

    Parameters:
    - dataset: list of instances (context, word, position, sense
    - window_size: int defining how many adjacent words to consider
    - n: size of ngram

    Returns:
    - ngrams: array of shape (n_samples, n_features) with n-gram features
    """
    ng = []
    for inst in dataset:

        p = inst.position
        context_words = [word for word, pos in inst.context]
        features = {}

        for i in range(p - window_size, p + window_size + 1):
            # Ensure ngram is in bounds of the context
            if i >= 0 and i + n <= len(context_words):
                features[f"ngram_{i}:{i+n}"] = '_'.join(context_words[i:i+n])
        ng.append(features)

    dvectorizer = DictVectorizer(sparse=False)
    ngrams = dvectorizer.fit_transform(ng)

    return ngrams

def concatenate_feature_vectors(feature_vectors):
    """
    Concatenates feature vectors along columns (axis=1).
    
    Parameters:
    - feature_vectors: features vectors list
    
    Returns:
    - concatenated_vector: single numpy array that concatenate all features.
    """
    # Ensure all feature vectors are dense numpy arrays before concatenation
    dense_feature_vectors = [fv.toarray() if isinstance(fv, csr_matrix) else fv for fv in feature_vectors]
    
    concatenated_vector = np.concatenate(dense_feature_vectors, axis=1)
    
    return concatenated_vector

def train_model(features, labels, features_id):
    """
    Trains a classifier (Multinomial NB) using cross-validation and computes scores.
    
    Parameters:
    - features: feature vector
    - labels: target labels
    - features_id: identifier of the current configuration of features
    
    Returns:
    - tuple: containing the classifier name, feature_id and mean values of performance metrics
    """
    classifier = MultinomialNB()
    stratified_split = StratifiedKFold(n_splits=5, shuffle=False)

    scoring_metrics = ['precision_micro', 'recall_micro', 'f1_micro', 'accuracy']
    scores = cross_validate(classifier, features, labels, cv=stratified_split, scoring=scoring_metrics)
    
    avg_precision = scores['test_precision_micro'].mean()
    avg_recall = scores['test_recall_micro'].mean()
    avg_f1 = scores['test_f1_micro'].mean()
    avg_accuracy = scores['test_accuracy'].mean()

    return ('Multinomial', features_id, avg_accuracy, avg_precision, avg_recall, avg_f1)

def get_interest_synsets():
    """
    Retrieves specific WordNet synsets for the noun 'interest' and processes their definitions.
    Combination of code from Laboratory 8.
    """
    mapping = {
        'interest_1': 'interest.n.01',
        'interest_2': 'interest.n.03',
        'interest_3': 'pastime.n.01',
        'interest_4': 'sake.n.01',
        'interest_5': 'interest.n.05',
        'interest_6': 'interest.n.04',
    }
    synsets = []

    for ss in wordnet.synsets('interest', 'n'):
        if ss.name() in mapping.values():
            defn = ss.definition()
            tags = preprocess(defn)  # assumendo che `preprocess` sia definita altrove
            toks = [l for w, l, p in tags]
            synsets.append((ss,toks))

    return synsets, mapping

def calculate_metrics(algorithm, hyps, refs, hyps_list, refs_list):
    """
    Calculates precision, recall, F-measure, and accuracy.
    
    Parameters:
    - algorithm: name of the algorithm used
    - hyps: hypothesis sets per sense
    - refs: reference sets per sense
    - hyps_list: list of predicted senses
    - refs_list: list of reference senses
    
    Returns:
    - tuple: containing averaged scores and dataframe of scores per sense.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    scores = pd.DataFrame()

    for cls in hyps.keys():
        p = precision(refs[cls], hyps[cls])
        r = recall(refs[cls], hyps[cls])
        f = f_measure(refs[cls], hyps[cls], alpha=1)
        s = len(refs[cls])

        curr_scores = pd.DataFrame({f'{algorithm}-Sense': [cls],
                                    f'{algorithm}-Prec': [p],
                                    f'{algorithm}-Rec': [r], 
                                    f'{algorithm}-Supp': [s],
                                    f'{algorithm}-Fm': [f]}) 
          
        scores = pd.concat([scores, curr_scores], ignore_index=True) 
    
    scores = scores.set_index(f'{algorithm}-Sense')   
    acc = round(accuracy(refs_list, hyps_list), 3)

    avg_precision = scores[f'{algorithm}-Prec'].mean()
    avg_recall = scores[f'{algorithm}-Rec'].mean()
    avg_fmeasure = scores[f'{algorithm}-Fm'].mean()
    
    return acc, avg_precision, avg_recall, avg_fmeasure, scores

def wsd_evaluation(dataset, algorithm):
    """
    Evaluates different Word Sense Disambiguation (WSD) algorithms.
    
    Parameters:
    - dataset: list of instances (context, word, position, sense)
    - algorithm: identifier of the WSD algorithm to be used
    
    Returns:
    - tuple: containing algorithm name, a dash for the features, scores and a dataframe with more detailed scores.
    """
    synsets, mapping = get_interest_synsets()

    # To calculate metrics
    refs = {k: set() for k in mapping.values()}
    hyps = {k: set() for k in mapping.values()}
    refs_list = []
    hyps_list = []

    for i, inst in enumerate(dataset):

        txt = [t[0] for t in inst.context]
        raw_ref = inst.senses[0] 
        hyp = None

        if algorithm == 'LeskSimp' or algorithm == 'LeskSimplified2':
            synsets = [ss for ss in wordnet.synsets('interest', pos='n') if ss.name() in mapping.values()]
            hyp = lesk_simplified(context_sentence=txt, ambiguous_word=txt[inst.position], synsets=synsets,).name()
                
        elif algorithm == 'LeskOr':
            hyp = original_lesk(context_sentence=txt, ambiguous_word=txt[inst.position], synsets=synsets).name()

        elif algorithm == 'LeskSimi':
            hyp = lesk_similarity(context_sentence=txt, ambiguous_word=txt[inst.position], synsets=synsets,).name()

        ref = mapping.get(raw_ref)
        
        refs[ref].add(i)
        hyps[hyp].add(i)
        refs_list.append(ref)
        hyps_list.append(hyp)

    acc, avg_prec, avg_rec, avg_fmeas, df = calculate_metrics(algorithm, hyps, refs, hyps_list, refs_list)
    return (algorithm, '-', acc, avg_prec, avg_rec, avg_fmeas), df

def check_features(features):
    """
    Shape and types of features check
    """
    print(f'Validating features:')
    fix_shape = None
    for feature in features:
        for name, vector in feature.items():
            print(f' - Checking Feature {name} -> Shape {vector.shape}, Type{type(vector)}')
            if fix_shape is None:
                fix_shape = vector.shape
            else:
                if vector.shape[0] != fix_shape[0] and vector.shape[1] != fix_shape[1]:
                    return False
    
    return True

def dataset_info(dataset):
    """
    Some dataset info
    """
    print(f'Dataset info:')
    print(f' - Size: {len(dataset)}')

    print(f'Senseval instance info:')
    print(f' - Senseval instance position:', dataset[0].position)
    print(f' - Senseval instance context:\n', dataset[0].context)
    print(f' - Senseval instance sense:', dataset[0].senses)