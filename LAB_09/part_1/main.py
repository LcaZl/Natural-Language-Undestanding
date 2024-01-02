# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":

    train_dataset, dev_dataset, test_dataset, vocab_len, lang = load_dataset()

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Parameters configuration common to all experiments
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
    }

    experiments = {

        'Experiment_1' : {
            'model_name':'RNN',
            'weight_path':f'bin/labRNN.pth',
            'optmz_type':'SGD',
            'optmz_learning_rate':0.44,
            'output_dropout': 0,
            'embedding_dropout': 0,
            'hidden_layer_size': 250,
            'embedded_layer_size': 350,
            **experiment_base
        },

        'Experiment_2': {
            'model_name':'LSTM',
            'weight_path':f'bin/LSTM.pth',
            'optmz_type':'SGD',
            'optmz_learning_rate':0.44,
            'output_dropout': 0,
            'embedding_dropout': 0,
            'hidden_layer_size': 250, 
            'embedded_layer_size': 350,
            **experiment_base
        }, 

        'Experiment_3': {
            'model_name':'LSTM',
            'weight_path':f'bin/LSTM_dropout.pth',
            'optmz_type':'SGD',
            'optmz_learning_rate':0.44, 
            'output_dropout': 0.11, #235
            'embedding_dropout': 0.11, 
            'hidden_layer_size': 250,
            'embedded_layer_size': 350, 
            **experiment_base
        },

        'Experiment_4': {
            'model_name':'LSTM',
            'weight_path':f'bin/LSTM_Adam.pth',
            'optmz_type':'Adam',
            'optmz_learning_rate':0.0001,
            'output_dropout': 0.116,
            'embedding_dropout': 0.116,
            'hidden_layer_size': 250,
            'embedded_layer_size': 350, 
            **experiment_base
        }
    }

    scores = execute_experiments(experiments)
    print('\nExperiments comparison:\n')
    print(tabulate(scores, headers='keys', tablefmt='grid', showindex=True))
