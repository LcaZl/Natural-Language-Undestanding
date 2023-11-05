# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import pandas as pd
import os
import torch.optim as optim
import tqdm
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold

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
                model = SBJ_Model(
                    hidden_size=parameters['hidden_layer_size'],
                    embedding_size=parameters['embedding_layer_size'],
                    output_size=parameters['output_size'],
                    vocab_size=parameters['subj_vocab_size'],
                    dropout=parameters['dropout'],
                    bidirectional=parameters['bidirectional'],                    
                )
                model.apply(init_weights)

                optimizer = optim.Adam(model.parameters(), lr = parameters['learning_rate'])
                report_slot, report_intent, train_losses, eval_losses = train_lm(model, parameters, optimizer)


def train_lm(model, parameters, optimizer):
    losses_train = []
    losses_eval = []
    best_loss = math.inf
    patience = 3

    for i in tqdm(range(0,parameters['epochs'])):
        loss = train_loop(parameters['train_loader'], optimizer, model, parameters)
        losses_train.append(np.asarray(loss).mean())

        if i % 5 == 0:

            eval_loss, _ = eval_loop(parameters['dev_loader'], model, parameters)
            losses_eval.append(eval_loss)

            if eval_loss < best_loss:
                best_loss = eval_loss
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    eval_loss, _ = eval_loop(parameters['test_loader'], model, parameters)
    losses_eval.append(eval_loss)

    return report_slot, report_intent, losses_train, losses_eval

def train_loop(data_loader, optimizer, criterion, model, clip=5):
    model.train()
    total_loss = 0
    total_tokens = 0

    for sample in data_loader:
        texts, labels, lengths = sample['text'], sample['label'], sample['length']
        optimizer.zero_grad()
        output = model(texts, lengths)
        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_tokens += len(labels)
        
    average_loss = total_loss / total_tokens
    return average_loss

def eval_loop(data_loader, criterion, model):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for sample in data_loader:
            texts, labels, lengths = sample['text'], sample['label'], sample['length']
            output = model(texts, lengths)
            loss = criterion(output, labels)

            total_loss += loss.item() * len(labels)
            total_tokens += len(labels)

    average_loss = total_loss / total_tokens
    return average_loss

