# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    folds, test_loader, lang = load_dataset()
    grid_search_parameters = {
        'hidden_layer_size': [250, 300, 350],
        'embedding_layer_size': [250, 300, 350],
        'learning_rate': [0.002, 0.001, 0.0005, 0.0001],
        'bidirectional': [True, False],
    }

    parameters = {
            'task': 'ABSA',
            'clip':5,
            'n_splits':10,
            'optimizer':'Adam',
            'dropout': 0.10,
            'learning_rate': 0.001,
            'grid_search':False,
            'grid_search_parameters': grid_search_parameters,
            'criterion':nn.CrossEntropyLoss(ignore_index = PAD_TOKEN),
            'lang':lang,
            'output_aspects': lang.aspect_labels,
            'output_polarities':lang.polarity_labels,
            'vocab_size': lang.vocab_size,
            'train_folds':folds,
            'test_loader':test_loader
        }
    
    model, training_report = train_model(parameters)
    print('\nOutput:\n',tabulate(training_report, headers='keys', tablefmt='grid', showindex=True))

    combined_report = pd.concat([training_report.sort_index()], axis=1)


    print('\nComaprison:\n',tabulate(combined_report, headers='keys', tablefmt='grid', showindex=True))
