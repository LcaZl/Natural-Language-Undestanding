# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
import pandas as pd
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from tabulate import tabulate
import torch.nn.init as init
from model import *
from utils import *
from evals import *

def grid_search(parameters):

    print('Starting grid search for:', parameters['grid_search_parameters'].keys(), '\n')
    grid = list(product(*parameters['grid_search_parameters'].values()))
    def to_parameter_dict(keys, values):
        return {key: value for key, value in zip(keys, values)}

    best_score = 0
    best_params = None
    best_model = None
    best_losses = None
    best_model_reports = None

    for i, params_tuple in enumerate(grid):
        combined_parameters = {**parameters, **to_parameter_dict(parameters['grid_search_parameters'].keys(), params_tuple)}

        print(f'({i+1}/{len(grid)})- Current parameters:',{key: combined_parameters[key] for key in combined_parameters.keys() if key in combined_parameters['grid_search_parameters'].keys()})

        b_model, reports, losses = train_lm(combined_parameters)
        
        average_f1 = np.mean([report[1] for report in reports])
        average_acc = np.mean([report[2] for report in reports])
        score = average_acc + average_f1

        print(f'- Average - Score F1: {average_f1} - Accuracy: {average_acc}')
        print(f'- Best model performance -  Score F1: {b_model[1][1]} - Accuracy: {b_model[1][2]}\n')
        
        if score > best_score:
            best_model_reports = reports
            best_params = combined_parameters
            best_model = b_model 
            best_losses = losses
            best_score = score

    print('\nEnd grid search:')
    print(f' - Best parameters: ',{key: combined_parameters[key] for key in best_params.keys() if key in best_params['grid_search_parameters'].keys()},'\n')

    return best_model, best_model_reports, best_losses, best_params

def init_weights(m):
    if isinstance(m, nn.RNN):
        # Initialize the RNN layers
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # Xavier initialization for input to hidden weights
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Orthogonal initialization for hidden to hidden weights
                init.orthogonal_(param.data)
            elif 'bias' in name:
                # Zero initialization for biases
                init.zeros_(param.data)
    elif isinstance(m, nn.Linear):
        # Kaiming initialization for the linear layers
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            # Zero initialization for the linear layer biases
            init.zeros_(m.bias)

def init_model(parameters, model_state = None):

    model = SUBJ_Model(
        hidden_size=parameters['hidden_layer_size'],
        embedding_size=parameters['embedding_layer_size'],
        output_size=parameters['output_size'],
        vocab_size=parameters['vocab_size'],
        dropout=parameters['dropout'],
        bidirectional=parameters['bidirectional'],                    
    ).to(DEVICE)
    if model_state:
        model.load_state_dict(model_state)
    else:
        model.apply(init_weights)

    if parameters['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), 
                    lr=parameters['learning_rate'])
    elif parameters['optimizer'] == 'Adam':
        optimizer = optim.AdamW(model.parameters(), 
                            lr=parameters['learning_rate'])

    return model, optimizer

def train_model(parameters):

    print(f'\nStart Training:')
    print(f'\n-------- ',parameters['task'],' --------\n')
    print('Parameters:\n')
    for key, value in parameters.items():
        if not key in ['train_folds', 'test_loader']:
            print(f' - {key}: {value}')
    print('\n')

    model_filename = f"models/{parameters['task']}_model.pth"

    if os.path.exists(model_filename):
        saved_data = torch.load(model_filename)
        print(f'Model founded. Parameters:', saved_data['parameters'])
        model, _ = init_model(parameters, saved_data['model_state'])
        reports = [saved_data['report']]
    else:
        if parameters['grid_search']:
            best_model, reports, losses, best_params = grid_search(parameters)
        else:
            best_model, reports, losses = train_lm(parameters)
           
        model = best_model[0]

        # Print losses
        for fold, losses in losses.items():
            print(f'Loss for {fold}:{losses}')

        # Save model and scores
        if (parameters['task'] != 'polarity_detection_with_filtered_dataset'):
            data_to_save = {
                'model_state': model.state_dict(),
                'report': best_model[1],
                'parameters':best_params
            }
            torch.save(data_to_save, model_filename)

    cols = ['Fold', 'F1-score', 'Accuracy']
    training_report = pd.DataFrame(list(reports), columns=cols).set_index('Fold')

    return model, training_report