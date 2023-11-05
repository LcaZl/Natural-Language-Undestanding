# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
"""
Create a pipeline model for Subjectivity and Polarity detection tasks. The pipeline has to be composed of two different models:

1. The first model predicts if a sentence is subjective or objective;
2. The second model performs the polarity detection of a document after removing the objective sentences predicted by the first model;

You have to report the results of the first and the second models. For the second model, you have to report the resutls achieved with and without the removal of the objective sentences to see if the pipeline can actually improve the performance.

The type of model: You have to choose a Neutral Network in PyTorch (e.g. MLP or RNN ) or a pre-trained language model (e.g. BERT or T5).

Datasets:
- NLTK: subjectivity (Subjectivity task)
- NLTK: movie reviews (Polarity task)

Evaluation:
Use a K-fold evaluation for both tasks where with K = 10

import nltk
nltk.download("subjectivity")
from nltk.corpus import movie_reviews
from nltk.corpus import subjectivity
print(len(subjectivity.sents()))

"""
from functions import *

if __name__ == "__main__":

    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    vocab_size = 10000
    test_size = 0.2 # Train, dev, test
    kfold = KFold(n_splits = 10, shuffle=True, random_state=42)

    print('\nLoading datasets ...\n')

    subj_fold_dataset, subj_test, subj_lang = load_subj_dataset(kfold, vocab_size, test_size)
    print('Subjectivity dataset folds (', len(subj_fold_dataset), '):')
    for k, fold in enumerate(subj_fold_dataset):
        print('- Fold',k,' dim -> Train:',len(fold[0]), 'Dev:', len(fold[1]))

    mr_fold_dataset, mr_test, mr_lang = load_mr_dataset(kfold, vocab_size, test_size)
    print('Movie reviews dataset folds (', len(mr_fold_dataset), '):')
    for sample in mr_fold_dataset:
        print('- Fold',k,' dim -> Train:',len(fold[0]), 'Dev:', len(fold[1]))

    experiments = {

        'Experiment_1':{
            'epochs':5,
            'runs':5,
            'learning_rate': 1e-4,
            'hidden_layer_size': 300,
            'embedding_layer_size' : 300,
            'output_size':2,
            'dropout': 0.1,
            'mr_vocab_size': mr_lang.vocab_size,
            'subj_vocab_size': subj_lang.vocab_size,
            'bidirectional':False,
            'movie_review_train':mr_fold_dataset,
            'movie_review_test':mr_test,
            'subj_train':subj_fold_dataset,
            'subj_test':subj_test,
            'movie_review_lang':mr_lang,
            'subj_lang':subj_lang
        }
    }

    
    execute_experiments(experiments)

