# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions

    train_dataset, dev_dataset, test_dataset, vocab_len, lang = load_dataset()

    train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    experiment_base = {
            'train_loader': train_loader,
            'dev_loader': dev_loader,
            'test_loader': test_loader,
            'vocab_len':vocab_len,
            'criterion_train':criterion_train,
            'criterion_eval':criterion_eval,
            'lang':lang,
            'n_epochs':100,
            'patience':3,
            'clip':5,
            'device':'cuda:0',
            'train_max_len': train_dataset.get_max_length(),
            'dev_max_len': dev_dataset.get_max_length(),
            'test_max_len':test_dataset.get_max_length()
    }

    experiments = {

        'Experiment_1': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_WeightTying.pth',
            'optmz_type':'Adam',
            'optmz_learning_rate':1e-3,
            'output_dropout': 0.10,
            'embedding_dropout': 0.10,
            'hidden_layer_size': 300,  
            'embedded_layer_size': 300, 
            'weight_tying': True,
            'variational_dropout': False,
            **experiment_base
        }, 

        'Experiment_2': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_VarDropout.pth',
            'optmz_type':'Adam',
            'optmz_learning_rate':1e-3,
            'output_dropout': 0.10,
            'embedding_dropout': 0.10,
            'hidden_layer_size': 300, 
            'embedded_layer_size': 300, 
            'weight_tying': True,
            'variational_dropout': True,
            **experiment_base
        },

        'Experiment_3': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_NTAvSGD.pth',
            'optmz_type':'NT-AvSGD',
            'optmz_learning_rate':0.01,
            'output_dropout': 0.10,
            'embedding_dropout': 0.10,
            'hidden_layer_size': 300, 
            'embedded_layer_size': 300,
            'weight_tying': True,
            'variational_dropout': True,
            'logging_interval': 1,
            'non_monotonic_interval': 5,
            **experiment_base
        }
    }

    scores = execute_experiments(experiments)
    print('\nExperiments comparison:\n')
    print(tabulate(scores, headers='keys', tablefmt='grid', showindex=True))
