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

def init_model(parameters, model_state = None):

    model = jointBERT(
        output_aspects=parameters['output_aspects'],
        output_polarities = parameters['output_polarities']
    ).to(DEVICE)

    if model_state:
        model.load_state_dict(model_state)

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
        #if parameters['grid_search']:
            #best_model, report, losses, best_params = grid_search(parameters)
        #else:
        best_model, report, losses = train_lm(parameters)
        best_params = parameters
        model = best_model[0]

        # Print losses
        for fold, losses in losses.items():
            print(f'Loss for {fold}:{losses}')

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
        aspect_logits, polarity_logits = model(sample['texts'], sample['attention_mask'], None)
        aspect_loss = parameters['criterion'](aspect_logits.view(-1, aspect_logits.shape[-1]), sample['y_aspects'].view(-1))
        polarity_loss = parameters['criterion'](polarity_logits.view(-1, polarity_logits.shape[-1]), sample['y_polarities'].view(-1))
        loss = aspect_loss + polarity_loss  # Considera di pesare diversamente le due loss se necessario
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
        optimizer.step()

    return round(np.mean(losses), 3)


def eval_loop(data_loader, model, parameters):
    model.eval()
    aspect_preds = []
    aspect_labels = []
    polarity_preds = []
    polarity_labels = []
    losses = []

    with torch.no_grad():
        for sample in data_loader:
            aspect_logits, polarity_logits = model(sample['texts'], sample['attention_mask'], sample['token_type_ids'])

            aspect_loss = parameters['criterion'](aspect_logits.view(-1, aspect_logits.shape[-1]), sample['y_aspects'].view(-1))
            polarity_loss = parameters['criterion'](polarity_logits.view(-1, polarity_logits.shape[-1]), sample['y_polarities'].view(-1))
            loss = aspect_loss + polarity_loss           
            losses.append(loss.item())
            
            # Calcola le probabilità e le previsioni per gli aspetti
            aspect_probs = torch.softmax(aspect_logits, dim=1)
            aspect_preds_batch = torch.argmax(aspect_probs, dim=1)
            aspect_preds.extend(aspect_preds_batch.cpu().numpy())
            aspect_labels.extend(sample['y_aspects'].view(-1).cpu().numpy())
            
            # Calcola le probabilità e le previsioni per la polarità
            polarity_probs = torch.softmax(polarity_logits, dim=1)
            polarity_preds_batch = torch.argmax(polarity_probs, dim=1)
            polarity_preds.extend(polarity_preds_batch.cpu().numpy())
            polarity_labels.extend(sample['y_polarities'].view(-1).cpu().numpy())

    # Calcola le metriche di valutazione
    all_preds = [parameters['lang'].id2label[id] for id in all_preds]
    all_labels = [parameters['lang'].id2label[id] for id in all_labels]

    report = evaluate(aspect_labels, polarity_labels, aspect_preds, polarity_preds) # (PRECISION, RECALL, F1)
    print(report)
    return round(np.mean(losses), 3), report


