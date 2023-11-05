# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
    training_set, validation_set, test_set, lang = load_dataset()
    
    # Dataloader instantiation
    train_loader = DataLoader(training_set, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(validation_set, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=64, collate_fn=collate_fn)
    
    experiment_base = {
            'model':'IAS',
            'train_loader':train_loader,
            'dev_loader':dev_loader,
            'test_loader':test_loader,
            'clip':5,
            'output_slots':len(lang.slot2id),
            'output_intent':len(lang.intent2id),
            'vocabulary_size':len(lang.word2id),
            'epochs':20,
            'patience':3,
            'lang':lang,
            'runs':5,
    }


    experiments = {
        'Experiment_1':{
            'hidden_layer_size':300,
            'embedded_layer_size':300,
            'learning_rate':0.001,
            'dropout_probabilities':0,
            'bidirectional':False,
            **experiment_base
        },
        
        'Experiment_2':{
            'hidden_layer_size':300,
            'embedded_layer_size':300,
            'learning_rate':0.001,
            'dropout_probabilities':0,
            'bidirectional': True,
            **experiment_base
        },

        'Experiment_3':{
            'hidden_layer_size':300,
            'embedded_layer_size':300,
            'learning_rate':0.001,
            'dropout_probabilities':0.05,
            'bidirectional': True,
            **experiment_base
        }
    }

    results = execute_experiments(experiments)
    print('\nExperiments comparison:\n')
    print(tabulate(results, headers='keys', tablefmt='grid', showindex=True))
