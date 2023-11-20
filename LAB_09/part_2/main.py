# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions

    train_dataset, dev_dataset, test_dataset, vocab_len, lang = load_dataset()

    train_loader = DataLoader(train_dataset, batch_size=40, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=40, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=40, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
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
            'layers':3,
            'output_dropout': 0.10,
            'embedding_dropout': 0.10,
            'hidden_layer_size': 400,  
            'embedded_layer_size': 400,             
            'train_max_len': train_dataset.get_max_length(),
            'dev_max_len': dev_dataset.get_max_length(),
            'test_max_len':test_dataset.get_max_length()
    }

    experiments = {
        'Experiment_0': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM.pth',
            'optmz_type':'Adam',
            'clip':5,
            'optmz_learning_rate': 5e-3,
            'weight_tying': False,
            'variational_dropout': False,
            **experiment_base
        }, 

        'Experiment_1': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_WT.pth',
            'optmz_type':'Adam',
            'clip':5,
            'optmz_learning_rate': 5e-3,
            'weight_tying': True,
            'variational_dropout': False,
            **experiment_base
        }, 

        'Experiment_2': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_VD.pth',
            'optmz_type':'Adam',
            'clip':5,
            'optmz_learning_rate': 5e-3,
            'weight_tying': False,
            'variational_dropout': True,
            **experiment_base
        }, 

        'Experiment_3': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_WTVD.pth',
            'optmz_type':'Adam',
            'clip':5,
            'optmz_learning_rate': 5e-3,
            'weight_tying': True,
            'variational_dropout': True,
            **experiment_base
        },

        'Experiment_4': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_NTAvSGD.pth',
            'optmz_type':'NT-AvSGD',
            'optmz_learning_rate':0.7,
            'weight_tying': False,
            'clip':5,
            'variational_dropout': False,
            'logging_interval': 1,
            'non_monotonic_interval': 5,
            **experiment_base
        },

        'Experiment_5': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_NTAvSGD_WT.pth',
            'optmz_type':'NT-AvSGD',
            'optmz_learning_rate':0.7,
            'clip':5,
            'weight_tying': True,
            'variational_dropout': False,
            'logging_interval': 1,
            'non_monotonic_interval': 5,
            **experiment_base
        },

        'Experiment_6': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_NTAvSGD_VD.pth',
            'optmz_type':'NT-AvSGD',
            'optmz_learning_rate':0.7,
            'clip':5,
            'weight_tying': False,
            'variational_dropout': True,
            'logging_interval': 1,
            'non_monotonic_interval': 5,
            **experiment_base
        },

        'Experiment_7': {
            'model_name':'LSTM',
            'weight_path':f'models_weight/LSTM_NTAvSGD_WTVD.pth',
            'optmz_type':'NT-AvSGD',
            'optmz_learning_rate':0.7,
            'clip':5,
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
