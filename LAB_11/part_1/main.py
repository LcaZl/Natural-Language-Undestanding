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
CUDA_LAUNCH_BLOCKING=1
if __name__ == "__main__":

    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    test_size = 0.1 # Train, dev, test
    FOLDS = 10
    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle = True)

    print('Loading datasets ...\n')

    subj_fold_dataset, subj_test, subj_lang = load_dataset('Subjectivity', skf, test_size)

    mr_fold_dataset, mr_test, mr_lang = load_dataset('Movie_reviews', skf, test_size)

    print('Datasets loaded.\n')

    grid_search_parameters = {
        'hidden_layer_size': [200, 250, 300],
        'embedding_layer_size': [200, 250, 300],
        'learning_rate': [0.001, 0.0001, 0.00005],
        'dropout':[0, 0.1]
    }

    training_baseline = {
            'clip':5,
            'n_splits':FOLDS,
            'epochs':200,
            'runs':3,
            'output_size':1,
            'criterion': nn.BCEWithLogitsLoss(),
            'optimizer':'Adam',
            'grid_search_parameters': grid_search_parameters,
            'vader_score': False
    }

    training_parameters = {
        'Subj_model':{
            **training_baseline,

            'task':'subjectivity_detection',
            'learning_rate': 5e-5, #0.0005
            'dropout':0.1,      

            'vocab_size': subj_lang.vocab_size,
            'train_folds':subj_fold_dataset,
            'test_loader':subj_test,
            'lang':subj_lang,
            'grid_search':False

        },
        'polarity_model':{
            **training_baseline,

            'task':'polarity_detection',
            'learning_rate': 5e-4,
            'dropout':0.1,         

            'vocab_size': mr_lang.vocab_size,
            'train_folds':mr_fold_dataset,
            'test_loader':mr_test,
            'lang':mr_lang,
            'grid_search':False

        }
    }

    subj_model, subj_training_report = train_model(training_parameters['Subj_model'])
    print('\nOutput:\n',tabulate(subj_training_report, headers='keys', tablefmt='grid', showindex=True))

    pol_model, pol_training_report = train_model(training_parameters['polarity_model'])
    print('\nOutput:\n',tabulate(pol_training_report, headers='keys', tablefmt='grid', showindex=True))

    mr4subj_fold_dataset, _, mr4subj_lang = load_dataset('movie_review_4subjectivity', skf, test_size, args = [subj_lang])

    print('\nFiltering sentences of movie reviews ...')
    filter = create_subj_filter(mr4subj_fold_dataset, subj_model, mr_lang, subj_lang)

    mr2_fold_dataset, mr2_test, mr2_lang = load_dataset('movie_review_filtered', skf, test_size, args = [filter])

    training_parameters['polarity_model_no_obj'] = {
        **training_baseline,
        
        'task':'polarity_detection_with_filtered_dataset',
        'learning_rate': 0.00075,
        'dropout':0.1,

        'vocab_size': mr2_lang.vocab_size,
        'train_folds':mr2_fold_dataset,
        'test_loader':mr2_test,
        'lang':mr2_lang, 
        'grid_search':False,

    }
    
    pol2_model, pol2_training_report = train_model(training_parameters['polarity_model_no_obj'])
    print('\nOutput:\n',tabulate(pol2_training_report, headers='keys', tablefmt='grid', showindex=True))


    # Rinomina le colonne per ciascun DataFrame per riflettere il modello
    subj_training_report.columns = [f'Subj_{col}' for col in subj_training_report.columns]
    pol_training_report.columns = [f'Pol_{col}' for col in pol_training_report.columns]
    pol2_training_report.columns = [f'Pol_No_Obj_{col}' for col in pol2_training_report.columns]

    # Unisci i DataFrame utilizzando l'indice 'Fold'
    combined_report = pd.concat([subj_training_report.sort_index(), pol_training_report.sort_index(), pol2_training_report.sort_index()], axis=1)
    print('\nComaprison:\n',tabulate(combined_report, headers='keys', tablefmt='grid', showindex=True))
