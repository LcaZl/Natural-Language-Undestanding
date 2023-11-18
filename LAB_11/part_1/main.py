from functions import *

if __name__ == "__main__":

    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    test_size = 0.1 # Train, dev, test
    FOLDS = 10
    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle = True)

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
    }

    training_parameters = {}

    # Training for subjectivity

    subj_fold_dataset, subj_test, subj_lang = load_dataset('Subjectivity', skf, test_size, tr_batch = 128, vl_batch = 64)

    training_parameters['Subj_model'] = {
            **training_baseline,

            'task':'subjectivity_detection',
            'learning_rate': 5e-5, #0.0005
            'dropout':0.05,      

            'vocab_size': subj_lang.vocab_size,
            'train_folds':subj_fold_dataset,
            'test_loader':subj_test,
            'lang':subj_lang,
            'grid_search':False

        }
    
    subj_model, subj_training_report = train_model(training_parameters['Subj_model'])
    print('\nOutput:\n',tabulate(subj_training_report, headers='keys', tablefmt='grid', showindex=True))

    # Training for polarity
    mr_fold_dataset, mr_test, mr_lang = load_dataset('Movie_reviews', skf, test_size, tr_batch = 80, vl_batch = 48)

    training_parameters['polarity_model'] = {
            **training_baseline,

            'task':'polarity_detection',
            'learning_rate': 5e-5,
            'dropout':0,         

            'vocab_size': mr_lang.vocab_size,
            'train_folds':mr_fold_dataset,
            'test_loader':mr_test,
            'lang':mr_lang,
            'grid_search':False

        }
    
    pol_model, pol_training_report = train_model(training_parameters['polarity_model'])
    print('\nOutput:\n',tabulate(pol_training_report, headers='keys', tablefmt='grid', showindex=True))

    # Training pipeline 

    mr4subj_fold_dataset, _, mr4subj_lang = load_dataset('movie_review_4subjectivity', skf, test_size, args = [subj_lang], tr_batch = 80, vl_batch = 48)

    print('\nCreating filter for movie reviews ...')
    filter = create_subj_filter(mr4subj_fold_dataset, subj_model, subj_lang)

    mr2_fold_dataset, mr2_test, mr2_lang = load_dataset('movie_review_filtered', skf, test_size, args = [filter], tr_batch = 80, vl_batch = 48)

    training_parameters['polarity_model_no_obj'] = {
        **training_baseline,
        
        'task':'polarity_detection_with_filtered_dataset',
        'learning_rate': 5e-5,
        'dropout':0.0,

        'vocab_size': mr2_lang.vocab_size,
        'train_folds':mr2_fold_dataset,
        'test_loader':mr2_test,
        'lang':mr2_lang, 
        'grid_search':False,

    }
    
    pol2_model, pol2_training_report = train_model(training_parameters['polarity_model_no_obj'])
    print('\nOutput:\n',tabulate(pol2_training_report, headers='keys', tablefmt='grid', showindex=True))


    # Final output
    subj_training_report.columns = [f'Subj_{col}' for col in subj_training_report.columns]
    pol_training_report.columns = [f'Pol_{col}' for col in pol_training_report.columns]
    pol2_training_report.columns = [f'Pol_No_Obj_{col}' for col in pol2_training_report.columns]

    combined_report = pd.concat([subj_training_report.sort_index(), pol_training_report.sort_index(), pol2_training_report.sort_index()], axis=1)
    print('\nComaprison:\n',tabulate(combined_report, headers='keys', tablefmt='grid', showindex=True))
