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

from utils import *
from model import *

REMOVE_CLASS = 'obj'

def create_subj_filter(dataset, model, lang, subj_lang):
    model.to(DEVICE)
    model.eval()
    filter = []
    
    with torch.no_grad():
        
        # Filtrare il train set
        for sample in tqdm(dataset):

            outputs = model(sample['text'], sample['attention_masks'])#, sample['vlabels'])
            predictions = torch.round(torch.sigmoid(outputs))
            subjective_mask = predictions.view(-1) == 1

            for i in range(sample['text'].size(0)):
                if subj_lang.id2class[subjective_mask.tolist()[i]] == REMOVE_CLASS:

                    decoded_text = lang.decode(sample['text'][i].tolist())[1:-1]
                    filter.append(decoded_text)

    return filter

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
        
        f = b_model[1][1]
        acc = b_model[1][2]
        score = (f + acc) / 2

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

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def init_model(parameters, model_state = None):

    model = SUBJ_Model(
        output_size=parameters['output_size'],
        dropout=parameters['dropout']
        #vader = parameters['vader_score']                   
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
            best_params = parameters
        model = best_model[0]

        # Print losses
        for fold, losses in losses.items():
            print(f'Loss for {fold}:{losses}')

        # Save model and scores
        if (parameters['task'] != 'polarity_detection_with_filtered_dataset'):
            data_to_save = {
                'model_state': model.state_dict(),
                'best_report': best_model[1],
                'parameters':best_params,
                'report':reports
            }
            torch.save(data_to_save, model_filename)

    cols = ['Fold','Run','F1-score', 'Accuracy']
    training_report = pd.DataFrame(reports, columns=cols).set_index('Fold')

    return model, training_report

def train_lm(parameters):
    cols = ['Fold','Run','F1-score', 'Accuracy']
    losses = {}
    reports = []
    best_score = 0

    for i in range(0,parameters['n_splits']):

        print(f'\nFOLD {i}:')
        train_loader, dev_loader = parameters['train_folds'][i]
        fold_reports = []

        score, report = None, None
        pbar = tqdm(range(0, parameters['runs']))
        for r in pbar:

            model, optimizer = init_model(parameters)
            loss_idx = f'Fold_{i}'
            losses[loss_idx] = []
            P = 3
            S = 0

            for epoch in range(0, parameters['epochs']):   

                loss = train_loop(train_loader, optimizer, model, parameters)
                losses[loss_idx].append(loss)

                if epoch % 2:
                    _, score, report = evaluation(model, parameters, dev_loader)
                    
                    if score > S:
                        S = score
                    else:
                        P -= 1

                    if P <= 0:
                        break

                pbar.set_description(f'Run {r} - Epoch {epoch} - L: {loss} - S:{score} - Report:{report}')

            _, score, report = evaluation(model, parameters, parameters['test_loader'])

            report = [i] + [r] + report
            reports.append(report)
            fold_reports.append(report)

            if score > best_score:
                best_score = score
                best_model = (model, report)

        fold_df = pd.DataFrame(fold_reports, columns=cols).set_index('Fold')
        print(tabulate(fold_df, headers='keys', tablefmt='grid', showindex=True))
        print(best_model[1])

    return best_model, reports, losses

def evaluation(model, parameters, dataset):

    loss, report = eval_loop(dataset, model, parameters)
    f = round(report['macro avg']['f1-score'], 4)
    acc = round(report['accuracy'], 4)
    score = round(np.mean([f, acc]), 4)
    report = [f, acc]
    return loss, score, report

def train_loop(data_loader, optimizer, model, parameters):
    model.train()
    losses = []

    for sample in data_loader:
        optimizer.zero_grad()

        input_ids = sample['text']
        attention_mask = sample['attention_masks']
        #vader_scores = sample['vlabels']

        output = model(input_ids, attention_mask)#, vader_scores)

        loss = parameters['criterion'](output.view(-1), sample['labels'].float())
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
            input_ids = sample['text']
            attention_mask = sample['attention_masks']
            #vader_scores = sample['vlabels']

            outputs = model(input_ids, attention_mask)#, vader_scores)
            loss = parameters['criterion'](outputs.view(-1), sample['labels'].float())
            losses.append(loss.item())

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(sample['labels'].cpu().numpy())

    report = classification_report(all_labels, all_preds, zero_division=False, output_dict=True)

    return round(np.mean(losses), 3), report

