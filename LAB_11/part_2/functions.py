# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
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
from itertools import product

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

    model = AspectTermExtractor(
        hidden_size=parameters['hidden_layer_size'],
        embedding_size=parameters['embedding_layer_size'],
        output_size=parameters['output_size'],
        vocab_size=parameters['vocab_size']
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
            best_model, report, losses, best_params = grid_search(parameters)
        else:
            best_model, report, losses = train_lm(parameters)
            best_params = parameters
        model = best_model[0]

        # Print losses
        for fold, losses in losses.items():
            print(f'Loss for {fold}:{losses}')

        # Save model and scores
        if (parameters['task'] != 'polarity_detection_with_filtered_dataset'):
            data_to_save = {
                'model_state': model.state_dict(),
                'report': report,
                'parameters': best_params 
            }
            torch.save(data_to_save, model_filename)

    cols = ['Fold', 'F1-score', 'Accuracy']
    training_report = pd.DataFrame(report, columns=cols).set_index('Fold')

    return model, training_report

def train_lm(parameters):
    losses = {}
    reports = []
    best_score = 0

    pbar = tqdm(range(0,parameters['n_splits']))
    for i in pbar:
        loss_idx = f'Fold_{i}'
        losses[loss_idx] = []
        model, optimizer = init_model(parameters)

        train_loader, dev_loader = parameters['train_folds'][i]

        loss = train_loop(train_loader, optimizer, model, parameters)
        losses[loss_idx].append(loss)

        loss, report = eval_loop(dev_loader, model, parameters)
        losses[loss_idx].append(loss)

        f = round(report[2], 3)
        recall = round(report[1], 3)
        prec = round(report[0], 3)

        if f > best_score:
            best_score = f
            best_model = model
        
        pbar.set_description(f'FOLD {i+1} - F1:{f} - P:{prec} - R:{recall}')

    loss, report = eval_loop(parameters['test_loader'], best_model, parameters)
    losses[loss_idx].append(loss)
    f = round(report[2], 3)
    recall = round(report[1], 3)
    prec = round(report[0], 3)
    report = (model, [i, f, prec, recall])
    return best_model, report, losses

def train_loop(data_loader, optimizer, model, parameters):
    model.train()
    losses = []

    for sample in data_loader:
        optimizer.zero_grad()
        # Il modello restituirà logits di dimensione [batch_size, seq_length, label_size]
        output = model(sample['text'], sample['lengths'])
        
        # Ridimensiona l'output e le etichette per calcolare la loss
        output = output.view(-1, output.shape[-1])  # Cambia la forma a [batch_size * seq_length, label_size]
        labels = sample['labels'].view(-1)  # Cambia la forma a [batch_size * seq_length]
        
        loss = parameters['criterion'](output, labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
        optimizer.step()

    return round(np.mean(losses), 3)


def eval_loop(data_loader, model, parameters):
    model.eval()
    all_preds = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for sample in data_loader:
            outputs = model(sample['text'], sample['lengths'])
            outputs = outputs.view(-1, outputs.shape[-1])  # Cambia la forma a [batch_size * seq_length, label_size]
            labels = sample['labels'].view(-1)  # Cambia la forma a [batch_size * seq_length]

            loss = parameters['criterion'](outputs, labels)
            losses.append(loss.item())
            
            # Calcola le probabilità e le predizioni
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Calcola le metriche di valutazione
    all_preds = [parameters['lang'].id2label[id] for id in all_preds]
    all_labels = [parameters['lang'].id2label[id] for id in all_labels]

    report = evaluate_ote(all_labels, all_preds) # (PRECISION, RECALL, F1)

    return round(np.mean(losses), 3), report