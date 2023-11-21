# Libraries
import numpy as np
import pandas as pd
import string
from collections import Counter
from tabulate import tabulate
from tqdm import tqdm


# Sklearn
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning


# text processing
import nltk
from nltk.stem import WordNetLemmatizer
from word2number import w2n
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')


def vectorize_dataset(dataset, parameters):
    """
    Vectorize the given dataset using either TF-IDF or Count Vectorizer method based on the input parameters.

    Parameters:
    - dataset (list): A list of documents to be vectorized.
    - parameters (dict): A dictionary of the parameters of current experiment.

    Returns:
    - matrix: Transformed dataset as a term-document matrix.
    """
    if parameters['vectorization_method'] == 'TF_IDF':

        vectorizer = TfidfVectorizer(
            strip_accents=parameters['strip_accents'],
            lowercase=parameters['lowercase'],
            stop_words=parameters['stop_words'],
            max_df=parameters['max_df'],
            min_df=parameters['min_df'],
            max_features=parameters['max_features']
        )

    elif parameters['vectorization_method'] == 'CountVect':

        vectorizer = CountVectorizer(
            strip_accents=parameters['strip_accents'],
            lowercase=parameters['lowercase'],
            stop_words=parameters['stop_words'],
            max_df=parameters['max_df'],
            min_df=parameters['min_df'],
            max_features=parameters['max_features']
        )


    dataset = vectorizer.fit_transform(dataset)

    return dataset

def execute_experiment(id, parameters, dataset, cv_methods, cv_folds, baseline_st, metrics):
    """
    Execute an experiment accordingly to the input parameters
    
    Parameters:
    - id (str): experiment id
    - parameters (dict): parameters of the current experiment vectorization
    - dataset (str): dataset to use
    - cv_methods (list): cross validation methods to use
    - cv_folds (int): number of cross validation folds
    - baseline_st (list): baseline strategies adopted
    - metrics (list): metrics to evaluate the model

    Returns:
    - df (pd.DataFrame): dataFrame rapresenting the experiment results.
    """
    for p_name, p_value in parameters.items():
        print(f' - {p_name} : {p_value}')

    # Dataset and output report initialization
    cr_dataset = dataset.copy() 
    reports_fold_lv = pd.DataFrame() # Contains scores for each fold and baseline type of current experiment.

    # Dataset vectorization (accordingly to current experiment parameters)
    cr_dataset['data'] = vectorize_dataset(cr_dataset['data'], parameters)
    
    # Train SVMs (one for each cross_validation method)
    print(f'\nTraining SVMs ...\n')

    for cv_method in cv_methods:
        
        # 0.002 with 1000 iter -> Ok (Fastest)
        # 0.003 with 1000 iter -> Fail
        # 0.004 with 1000 iter -> Fail
        # 0.003 with 1500 iter -> Ok (Slower and less accurate)
        # 0.0025 with 1000 iter -> Fail
        # 0.0025 with 1250 iter -> Ok (Slowest)

        report_fold_lv, report_exp_lv = train_SVM(
            C = 0.0025, # Too high overfitting, Too low underfitting (Low C higher margin)
            iterations = 1350,
            dataset = cr_dataset, 
            folds = cv_folds, 
            cv_type = cv_method,
            baseline_strategies = baseline_st,
            scores=metrics
        )

        # Building output
        report_exp_lv = report_exp_lv.rename(id)
        reports_fold_lv = pd.concat([reports_fold_lv, report_fold_lv])

    # Current experiment results
    print_results(reports_fold_lv, f' - {id} results:\n')
    
    return reports_fold_lv, report_exp_lv

def train_SVM(dataset, folds, cv_type, baseline_strategies, scores, C = 1, iterations = 1000):
    """
    Train a SVM model on the given dataset and evaluate its performance using cross-validation.
    
    Parameters:
    - dataset (dict): Newsgroup dataset.
    - folds (int): Number of cross-validation folds.
    - cv_type (str): Type of cross-validation to use. Can be 'kfold' or 'skfold'.
    - baseline_strategies (list): A list of dummy classifier strategies to use as baselines.
    - scores (list): Metrics to evaluate the performance of the model
    - C (float, optional): Regularization parameter for the SVM. Default is 1.
    - iterations (int, optional): Maximum number of iterations for the SVM. Default is 1000.

    Returns:
    - df (pd.DataFrame): DataFrame containing the evaluation scores for each fold and the baselines.
    """

    # Create an SVM model with standard scaling. 'dual' is set to 'auto', which means
    # choose the dual formulation only when the number of samples is larger than the number of features
    svm_model = make_pipeline(preprocessing.StandardScaler(with_mean = False), 
                              LinearSVC( 
                                dual = 'auto',
                                C = C,
                                verbose = False,
                                max_iter = iterations)
                )
    
    # Get the cross-validation object based on the specified type
    cv = get_cv(cv_type, folds)

    # Compute the performance scores for baseline models
    baselines = get_model_baselines(baseline_strategies, 
                                    dataset, 
                                    cv, 
                                    scores)
    

    scores = cross_validate(svm_model, 
                            dataset['data'], 
                            dataset['target'], 
                            cv=cv, 
                            scoring=scores,
                            verbose = True)

    # Extract and format the evaluation scores into a DataFrame  
    cols = ['Accuracy','F1 Weighted','Recall Weighted']
    df = pd.DataFrame(data= zip(scores['test_accuracy'], 
                                scores['test_f1_weighted'],
                                scores['test_recall_weighted']),
                      columns = cols,
                      index = [f'{cv_type} - Fold {i}' for i, _ in enumerate(scores['test_accuracy'])])
    df_avg = df.mean()
    
    for baseline_name, baseline_value in baselines.items():
        df.loc[f'{cv_type} - {baseline_name}',] = baseline_value
    
    return df, df_avg
  
def get_cv(cv_type, folds):
    """
    Returns a cross-validation splitter based on the specified type.
    
    Parameters:
    - cv_type (str): Type of cross-validation to return
    - folds (int): Number of folds for the cross-validation.
    
    Returns:
    - Cross-validation splitter: An instance of either KFold or StratifiedKFold.
    """
    if cv_type == 'kfold':
        return KFold(n_splits=folds, shuffle=True, random_state = 10)
    else:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state = 10)

     
def get_model_baselines(types, dataset, cv, model_metrics):
    """
    Calculate baseline scores for the dataset using various dummy classifier strategies.
    
    Parameters:
    - types (list): Dummy classifier strategies.
    - dataset (dict): Dataset
    - cv (Cross-validation splitter): Cross-validation splitting strategy.
    - model_metrics (list): Model scores to compute.
    
    Returns:
    - all_scores (dict): Dictionary mapping each baseline type to its average accuracy.
    """
    all_scores = {}

    for type in types:
        dummy_clf = make_pipeline(preprocessing.StandardScaler(with_mean = False), 
                                  DummyClassifier(strategy = type))

        scores = cross_validate(dummy_clf, dataset['data'], dataset['target'], cv=cv, scoring=model_metrics)
        all_scores.update({f'Baseline-{type}' : np.mean(scores['test_accuracy'])})
        
    return all_scores

def dataset_info(dt):
    """
    Print a summary of the given dataset, including total samples, number of classes, and samples per class.
    
    Parameters:
    - dt: Newsgroup dataset.
    """
    print(f'\nNewsgroup dataset info:')
    print(f' - Samples: {len(dt.data)}')
    print(f' - Classes: {len(list(dt.target_names))}')
    print(f' - Samples per Class')
    labels = dt.target
    class_counts = Counter(labels)
    
    for class_label, count in class_counts.items():
        class_name = dt.target_names[class_label]
        print(f'   Class: {class_name}, Samples: {count}')
 
def print_results(df, title = None):
    """
    Prints the provided dataframe in a tabulated format.
    
    Parameters:
    - stats (DataFrame): Dataframe
    - title (str, optional): Title or header to be displayed above the statistics. 

    """
    
    if title is not None:
        print(f'{title}')
    print(tabulate(df, headers = "keys",tablefmt='simple_grid', floatfmt=".5f"))
    

def normalize_document(doc):
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
    - doc (String): document to normalize

    Returns:
    - list: document normalized

    """

    # Tokenize the document
    words = word_tokenize(doc)

    # Create a lemmatizer and stemmer instance
    lemmatizer = WordNetLemmatizer()

    # Remove newline characters
    words = [str(token).replace('\n', ' ') for token in words]
    
    # Remove punctuation
    words = [''.join(c for c in str(token) if c not in string.punctuation) for token in words]     
       
    # Number normalization (number words to numbers)
    words = [str(w2n.word_to_num(token)) if str(token) in w2n.american_number_system else token for token in words]
    
    # Lemmatization - Group togheter inflected forms of a word, so it can be analyzed as a single item
    words = [lemmatizer.lemmatize(str(token)) for token in words]
    
    # Rmove empty strings or whitespace words
    words = [token for token in words if token.strip()]
    
    return ' '.join(words)