from functions import *

if __name__ == "__main__":
    
    test_size = 0.1
    FOLDS = 10
    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle = True)
    best_models = pd.DataFrame(columns=['Experiment ID','Fold','Run','F-Score','Accuracy']).set_index('Experiment ID')
    training_parameters = {}
    training_baseline = {
            'clip':5,
            'n_splits':FOLDS,
            'epochs':200,
            'runs':3,
            'output_size':1,
            'criterion': nn.BCEWithLogitsLoss(),
            'optimizer':'Adam',
    }

    # Training for subjectivity -----------------------------------------------------------------------------------------------------------------------
    subj_fold_dataset, subj_test, subj_lang = load_dataset('Subjectivity', skf, test_size, tr_batch = 64, vl_batch = 32)
    training_parameters['Subj_model'] = {
            **training_baseline,
            'task':'subjectivity_detection',
            'learning_rate': 1e-4, #0.0005
            'dropout':0.2,      
            'vocab_size': subj_lang.vocab_size,
            'train_folds':subj_fold_dataset,
            'test_loader':subj_test,
            'lang':subj_lang,
        }
    
    subj_model, subj_training_report, best_report, losses = train_model(training_parameters['Subj_model'])

    run_level, fold_level = get_scores(subj_training_report)
    best_models.loc['subjectivity_detection'] = best_report
    print('\n - Run level metrics:\n',tabulate(run_level, headers='keys', tablefmt='grid', showindex=True))
    print('\n - Fold level metrics:\n',tabulate(fold_level, headers='keys', tablefmt='grid', showindex=True))
    print('\n - Best models:\n',tabulate(best_models, headers='keys', tablefmt='grid', showindex=True))
    plot_aligned_losses(losses[0][f'Fold_{best_report[0]}-run_{best_report[1]}'], 
                        losses[1][f'Fold_{best_report[0]}-run_{best_report[1]}'], 
                        'Subjectivity detection - Best model losses')

    # Training for polarity only ----------------------------------------------------------------------------------------------------------------------
    mr_fold_dataset, mr_test, mr_lang = load_dataset('Movie_reviews', skf, test_size, tr_batch = 32, vl_batch = 16)
    training_parameters['polarity_model'] = {
            **training_baseline,
            'task':'polarity_detection',
            'learning_rate': 5e-5,
            'dropout':0.2,         
            'vocab_size': mr_lang.vocab_size,
            'train_folds':mr_fold_dataset,
            'test_loader':mr_test,
            'lang':mr_lang,
        }
    
    pol_model, pol_training_report, best_report, losses = train_model(training_parameters['polarity_model'])
    
    run_level, fold_level = get_scores(pol_training_report)
    best_models.loc['polarity_detection'] = best_report
    print('\n - Run level metrics:\n',tabulate(run_level, headers='keys', tablefmt='grid', showindex=True))
    print('\n - Fold level metrics:\n',tabulate(fold_level, headers='keys', tablefmt='grid', showindex=True))
    print('\n - Best models:\n',tabulate(best_models, headers='keys', tablefmt='grid', showindex=True))
    plot_aligned_losses(losses[0][f'Fold_{best_report[0]}-run_{best_report[1]}'], 
                        losses[1][f'Fold_{best_report[0]}-run_{best_report[1]}'], 
                        'Polarity detection - Best model losses')
    
    # Filtering movie reviews -------------------------------------------------------------------------------------------------------------------------

    mr4subj_fold_dataset, _, mr4subj_lang = load_dataset('movie_review_4subjectivity', skf, test_size, args = [subj_lang], tr_batch = 32, vl_batch = 16)

    print('\nCreating filter for movie reviews ...')
    filter = create_subj_filter(mr4subj_fold_dataset, subj_model, subj_lang)

    mr2_fold_dataset, mr2_test, mr2_lang = load_dataset('movie_review_filtered', 
                                                        skf, test_size, 
                                                        args = [filter], 
                                                        tr_batch = 32, 
                                                        vl_batch = 16)
    
    # Training for polarity with filtered dataset  ----------------------------------------------------------------------------------------------------

    training_parameters['polarity_model_no_obj'] = {
        **training_baseline,
        'task':'polarity_detection_with_filtered_dataset',
        'learning_rate': 1e-4,
        'dropout':0.2,
        'vocab_size': mr2_lang.vocab_size,
        'train_folds':mr2_fold_dataset,
        'test_loader':mr2_test,
        'lang':mr2_lang, 
    }
    
    pol2_model, pol2_training_report, best_report, losses = train_model(training_parameters['polarity_model_no_obj'])

    run_level, fold_level = get_scores(pol2_training_report)
    best_models.loc['polarity_detection_no_objective'] = best_report
    print('\n - Run level metrics:\n',tabulate(run_level, headers='keys', tablefmt='grid', showindex=True))
    print('\n - Fold level metrics:\n',tabulate(fold_level, headers='keys', tablefmt='grid', showindex=True))
    print('\n - Best models:\n',tabulate(best_models, headers='keys', tablefmt='grid', showindex=True))
    plot_aligned_losses(losses[0][f'Fold_{best_report[0]}-run_{best_report[1]}'], 
                        losses[1][f'Fold_{best_report[0]}-run_{best_report[1]}'], 
                        'Polarity detection with filtered dataset - Best model losses')