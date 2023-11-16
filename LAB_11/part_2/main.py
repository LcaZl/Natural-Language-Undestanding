# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
    
if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle = True)
    folds, test_loader, lang = load_dataset(skf)

    parameters = {
            'task': 'ABSA',
            'clip':5,
            'epochs': 200,
            'n_splits':10,
            'runs':5,
            'optimizer':'Adam',
            'dropout': 0.1,
            'learning_rate': 5e-5,
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
