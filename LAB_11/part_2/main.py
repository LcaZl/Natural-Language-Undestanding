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
            'optimizer':'Adam',
            'dropout': 0.05,
            'learning_rate': 1e-4,
            'lang':lang,
            'output_aspects': lang.aspect_labels,
            'output_polarities':lang.polarity_labels,
            'vocab_size': lang.vocab_size,
            'train_folds':folds,
            'test_loader':test_loader
        }
    

    model, training_report, best_report = train_model(parameters)

    print(f'Training report:\nBest model at fold {best_report[0]}.')
    print(tabulate(training_report.sort_index(), headers='keys', tablefmt='grid', showindex=True))
