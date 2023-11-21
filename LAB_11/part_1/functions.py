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
import json
import matplotlib.pyplot as plt

from utils import *
from model import *

REMOVE_CLASS = 'obj'

def get_scores(reports):
    df_runs = pd.DataFrame(columns=['Fold', 'Run', 'F1-score', 'F1 std.', 'Accuracy', 'Accuracy std.'])
    fs, accs = [], []

    for [fold, run, f, acc] in reports:
        fs.append(f)
        accs.append(acc)
        df_runs.loc[len(df_runs)] = [fold ,run, f, 0, acc, 0]
    
    df_runs = df_runs.round(4)

    df_folds = df_runs.groupby('Fold').agg({
        'F1-score': ['mean', 'std'],
        'Accuracy': ['mean', 'std']
    }).reset_index()
    df_folds.columns = ['Fold', 'F1-score Mean', 'F1-score Std', 'Accuracy Mean', 'Accuracy Std']

    df_folds = df_folds.round(4)
    return df_runs, df_folds

def train_model(parameters):

    print(f'\nStart Training:')
    print(f'\n-------- ',parameters['task'],' --------\n')
    print('Parameters:\n')

    # Print parameters
    for key, value in parameters.items():
        if not key in ['train_folds', 'test_loader']:
            print(f' - {key}: {value}')
    print('\n')

    model_filename = f"models/{parameters['task']}_model.pth"

    if os.path.exists(model_filename): # Model founded
        saved_data = torch.load(model_filename)
        print(f'Model founded. Parameters:', saved_data['parameters'])
        model, _ = init_model(saved_data['parameters'], saved_data['model_state'])
        reports = saved_data['report']
        best_report = saved_data['best_report']
        losses = saved_data['losses']

    else:
        best_model, reports, losses = train_lm(parameters)

        model = best_model[0]
        best_report = best_model[1]

        data_to_save = {
            'model_state': best_model[0].state_dict(),
            'best_report': best_model[1],
            'parameters':parameters,
            'report':reports,
            'losses':losses
        }
        torch.save(data_to_save, model_filename)

    return model, reports, best_report, losses

def init_model(parameters, model_state = None):

    model = SUBJ_Model(
        output_size=parameters['output_size'],
        dropout=parameters['dropout']
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

def train_lm(parameters):
    cols = ['Fold','Run','F1-score', 'Accuracy']
    dev_losses = {}
    train_losses = {}

    reports = []
    best_score = 0

    for i in range(0,parameters['n_splits']):

        train_loader, dev_loader = parameters['train_folds'][i]
        fold_reports = []

        dev_loss, score, report = None, None, None

        for r in range(0, parameters['runs']):
            print(f'\nFOLD {i} - Run {r}:')

            model, optimizer = init_model(parameters)
            loss_idx = f'Fold_{i}-run_{r}'
            train_losses[loss_idx], dev_losses[loss_idx] = [], []
             
            P = 4
            S = 0
            pbar = tqdm(range(0, parameters['epochs']))

            for epoch in pbar:   

                tr_loss = train_loop(train_loader, optimizer, model, parameters)

                if epoch % 1 == 0:
                    dev_loss, score, report = evaluation(model, parameters, dev_loader)
                    dev_losses[loss_idx].append(np.mean(dev_loss))
                    train_losses[loss_idx].append(np.mean(tr_loss))

                    if score > S:
                        S = score
                        P = 4
                    else:
                        P -= 1

                    if P <= 0:
                        break

                pbar.set_description(f'Epoch {epoch} - TL: {round(np.mean(tr_loss), 3)} - DL: {round(np.mean(dev_loss), 3)} - S:{score} - F1:{report[0]} - Acc:{report[1]}')

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
        
    return best_model, reports, (train_losses, dev_losses)

def train_loop(data_loader, optimizer, model, parameters):
    model.train()
    losses = []

    for sample in data_loader:
        optimizer.zero_grad()

        input_ids = sample['text']
        attention_mask = sample['attention_masks']

        output = model(input_ids, attention_mask)

        loss = parameters['criterion'](output.view(-1), sample['labels'].float())
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
        optimizer.step()

    return losses

def evaluation(model, parameters, dataset):

    losses, report = eval_loop(dataset, model, parameters)
    f = round(report['macro avg']['f1-score'], 4)
    acc = round(report['accuracy'], 4)
    score = round(f, 4)
    report = [f, acc]
    return losses, score, report

def eval_loop(data_loader, model, parameters):
    model.eval()
    all_preds = []
    all_labels = []    
    losses = []

    with torch.no_grad():
        for sample in data_loader:
            input_ids = sample['text']
            attention_mask = sample['attention_masks']

            outputs = model(input_ids, attention_mask)
            loss = parameters['criterion'](outputs.view(-1), sample['labels'].float())
            losses.append(loss.item())

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(sample['labels'].cpu().numpy())

    report = classification_report(all_labels, all_preds, zero_division=False, output_dict=True)

    return losses, report

def create_subj_filter(dataset, model, subj_lang, filename='subj_filter.json'):

    if os.path.exists(filename):
        with open(filename, 'r') as file:
            filter = json.load(file)
            print(f' - Filter found and loaded. Length:{len(filter)}.')
        return filter

    model.to(DEVICE)
    model.eval()
    filter = []

    with torch.no_grad():
        for sample in tqdm(dataset):
            outputs = model(sample['text'], sample['attention_masks'])
            predictions = torch.round(torch.sigmoid(outputs))
            subjective_mask = predictions.view(-1) == 0  # Id 0 means objective sentence
            
            for i in range(sample['text'].size(0)):
                if subj_lang.id2class[subjective_mask.tolist()[i]] == REMOVE_CLASS:
                    # Rimuovi CLS, SEP e padding
                    text_ids = sample['text'][i].tolist()
                    clean_text_ids = [id for id in text_ids if id != PAD_TOKEN]
                    filter.append(clean_text_ids)

    # Salva il filtro nel file
    with open(filename, 'w') as file:
        json.dump(filter, file)

    return filter

def plot_aligned_losses(training_losses, dev_losses, title):
    step = len(training_losses) / len(dev_losses)
    selected_training_losses = [training_losses[int(i * step)] for i in range(len(dev_losses))]

    # Crea il grafico
    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.plot(selected_training_losses, label='Training Loss')
    plt.plot(dev_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()