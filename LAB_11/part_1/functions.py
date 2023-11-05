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

from utils import *
from model import *

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.RNN):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            if m.bias:
                nn.init.zeros_(m.bias_ih_l0)
                nn.init.zeros_(m.bias_hh_l0)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            nn.init.zeros_(m.bias)

def execute_experiments(experiments_parameters):

    cols = []
    scores = pd.DataFrame(columns = cols)

    for exp_id, parameters in experiments_parameters.items():
        print(f'\n-------- {exp_id} --------\n')
        print('Parameters:\n')
        for key, value in parameters.items():
            if not key in ['movie_review_train', 'movie_review_test', 'subj_train','subj_test']:
                print(f' - {key}: {value}')

        print(f'\nStart Training:')
        measures = []
        for run in range(parameters['runs']):
            print(f'- Run {run}')
            model_filename = f"models_weight/{exp_id}.pth"

            if os.path.exists(model_filename):
                pass
            else:
                model = SUBJ_Model(
                    hidden_size=parameters['hidden_layer_size'],
                    embedding_size=parameters['embedding_layer_size'],
                    output_size=parameters['output_size'],
                    vocab_size=parameters['subj_vocab_size'],
                    dropout=parameters['dropout'],
                    bidirectional=parameters['bidirectional'],                    
                ).to(DEVICE)
                model.apply(init_weights)

                optimizer = optim.Adam(model.parameters(), lr = parameters['learning_rate'])
                parameters['model'] = 'SUBJ'
                report, tr_losses, eval_losses = train_lm(model, parameters, optimizer)
                print('Report:', report)
                print(tr_losses, eval_losses)

def train_lm(model, parameters, optimizer):
    losses_train = []
    losses_eval = []
    best_loss = math.inf
    patience = 3

    for i in tqdm(range(0,parameters['n_splits'])):

        if parameters['model'] == 'SUBJ':
            train_loader, dev_loader = parameters['subj_train_folds'][i]
        else:
            train_loader, dev_loader = parameters['mr_train_folds'][i]

        tr_loss = train_loop(train_loader, optimizer, model, parameters)

        losses_train.append(tr_loss)

        if i % 5 == 0:

            eval_loss, report = eval_loop(dev_loader, model, parameters)
            losses_eval.append(eval_loss)

            if eval_loss < best_loss:
                best_loss = eval_loss
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    if parameters['model'] == 'SUBJ':
        eval_loss, report = eval_loop(parameters['subj_test_loader'], model, parameters)
    else:
        eval_loss, report = eval_loop(parameters['mr_test_loader'], model, parameters)

    losses_eval.append(eval_loss)

    return report, losses_train, losses_eval

def train_loop(data_loader, optimizer, model, parameters):
    model.train()
    losses = []

    for sample in data_loader:
        texts, labels, lengths = sample['text'], sample['labels'], sample['lengths']
        print('TRAINLOOP:',sample['text'].shape, sample['labels'].shape, sample['lengths'].shape)

        # Assicurati che le dimensioni di labels siano corrette.
        # labels deve essere un vettore 1D con il valore di classe per ciascun elemento nel batch.

        optimizer.zero_grad()
        
        # Assicurati che le dimensioni di output siano corrette.
        # output deve essere di forma (batch_size, num_classes)
        output = model(texts, lengths)

        print('MODELOUTPUSHAPE:', output.shape)
        
        # Calcola la loss
        loss = parameters['criterion'](output, labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])

        optimizer.step()

    return np.mean(losses)

def eval_loop(data_loader, model, parameters):
    model.eval()
    all_preds = []
    all_labels = []    
    losses = []

    with torch.no_grad():
        for sample in data_loader:
            texts, labels, lengths = sample['text'], sample['labels'], sample['lengths']
            outputs = model(texts, lengths)            
            loss = parameters['criterion'](outputs, labels)

            losses.append(loss.item())

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=['Objective', 'Subjective'])

    return np.mean(losses), report

