from functions import *
    
if __name__ == "__main__":

    FOLDS = 10
    skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle = True)
    folds, test_loader, lang = load_dataset(skf)

    parameters = {
            'task': 'ABSA',
            'clip':5,
            'epochs': 200,
            'runs':5,
            'folds':FOLDS,
            'dropout': 0.2,
            'learning_rate': 5e-5,
            'lang':lang,
            'output_aspects': lang.aspect_labels,
            'output_polarities':lang.polarity_labels,
            'vocab_size': lang.vocab_size,
            'train_folds':folds,
            'test_loader':test_loader,
            'polarity_loss_coeff':0.75,
            'aspect_loss_coeff':0.25
        }
    
    model, reports, best_report, losses = experiment(parameters)

    # Result presentation
    cols = ['Fold', 'Run', 'ot_precision', 'ot_recall', 'ot_f1', 'ts_macro_f1', 'ts_micro_p', 'ts_micro_r', 'ts_micro_f1']
    metrics = ['ot_precision', 'ot_recall', 'ot_f1', 'ts_micro_p', 'ts_micro_r', 'ts_micro_f1'] #'ts_macro_f1'

    # Run level
    training_report_run_lv = pd.DataFrame(reports, columns=cols).set_index('Fold').round(3)
    print('\n - Run level metrics:\n',tabulate(training_report_run_lv.sort_index(), headers='keys', tablefmt='grid', showindex=True))

    # Fold level
    training_report_fold_lv = training_report_run_lv.groupby('Fold')[metrics].agg(['mean', 'std']).reset_index().round(3)
    training_report_fold_lv.columns = ['Fold'] + [f'{metric}_{stat}' if stat == 'Std' else f'{metric}' for metric in metrics for stat in ['Mean', 'Std']]
    print('\n - Fold level metrics:\n',tabulate(training_report_fold_lv.sort_index(), headers='keys', tablefmt='grid', showindex=True))

    # Best model scores and chart
    best_report_df = pd.DataFrame([best_report], columns=cols).set_index('Fold').round(3)
    print(f'\n - Best model at fold {best_report[0]}.')
    print(' - Model report:\n',tabulate(best_report_df.sort_index(), headers='keys', tablefmt='grid', showindex=True))
    plot_aligned_losses(losses[0][f'fold_{best_report[0]}-run_{best_report[1]}'], 
                    losses[1][f'fold_{best_report[0]}-run_{best_report[1]}'], 
                    'Best model losses')