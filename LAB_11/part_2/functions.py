import pandas as pd
import numpy as np
import os
from itertools import product
import math
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.init as init
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold
from evals import *

from model import *
from utils import *

def experiment(parameters):

    print(f'\nStart Training:')
    print(f'\n-------- ',parameters['task'],' --------\n')
    print('Parameters:\n')
    for key, value in parameters.items():
        if not key in ['train_folds', 'test_loader']:
            print(f' - {key}: {value}')
    print('\n')

    # Load model from file
    model_filename = f"bin/{parameters['task']}_model.pth"

    if os.path.exists(model_filename):

        saved_data = torch.load(model_filename)
        print(f'Model founded. \nParameters:', saved_data['parameters'])

        model, _ = init_model(parameters, saved_data['model_state'])
        reports = saved_data['reports']
        best_report = saved_data['best_report']
        losses = saved_data['losses']

    else:

        best_model, reports, losses = train_lm(parameters)
        best_params = parameters
        model = best_model[0]
        best_report = best_model[1]

        data_to_save = {
            'model_state': model.state_dict(),
            'reports': reports,
            'best_report': best_report,
            'parameters': best_params,
            'losses':losses
        }
        torch.save(data_to_save, model_filename)

    return model, reports, best_report, losses

def init_model(parameters, model_state = None):

    model = AspectSentimentModel(
        num_aspect_labels=parameters['output_aspects'],
        num_polarity_labels = parameters['output_polarities'],
        dropout_rate=parameters['dropout']
    ).to(DEVICE)

    if model_state:
        model.load_state_dict(model_state)

    optimizer = optim.AdamW(model.parameters(), 
                        lr=parameters['learning_rate'])

    return model, optimizer

def evaluation(model, parameters, dataset):
    losses, ote_report, ts_report = eval_loop(dataset, model, parameters)
    score = round(round(ts_report[0], 3), 2)
    report = [round(el, 3) for el in (ote_report + ts_report)]

    return losses, score, report

def train_lm(parameters):
    dev_losses = {}
    train_losses = {}
    reports = []
    best_score = 0
    
    cols = ['Fold', 'Run', 'ot_precision', 'ot_recall', 'ot_f1', 'ts_macro_f1', 'ts_micro_p', 'ts_micro_r', 'ts_micro_f1']

    for i in range(0, len(parameters['train_folds'])):

        train_loader, dev_loader, asp_weight, pol_weight = parameters['train_folds'][i]
        fold_reports = []

        parameters['asp_criterion'] = nn.CrossEntropyLoss(weight = torch.tensor(asp_weight).to(DEVICE))
        parameters['pol_criterion'] = nn.CrossEntropyLoss(weight = torch.tensor(pol_weight).to(DEVICE))
        
        for r in range(0, parameters['runs']):
            print(f'\nFOLD {i} - Run {r}:')

            dev_loss, score, report = None, None, None
            model, optimizer = init_model(parameters)
            loss_idx = f'fold_{i}-run_{r}'
            train_losses[loss_idx], dev_losses[loss_idx] = [], []

            P = 5
            S = 0
            pbar = tqdm(range(0,parameters['epochs']))

            for epoch in pbar:        
                tr_loss = train_loop(train_loader, optimizer, model, parameters)

                if epoch % 2 == 0:
                    dev_loss, score, report = evaluation(model, parameters, dev_loader)
                    dev_losses[loss_idx].append(np.mean(dev_loss))   
                    train_losses[loss_idx].append(np.mean(tr_loss))
  
                    if score > S:
                        S = score
                        P = 5
                    else:
                        P -= 1
                                    
                pbar.set_description(f'Epoch {epoch} - TL: {round(np.mean(tr_loss), 3)} - DL: {round(np.mean(dev_loss), 3)} - S:{score} - OTE:{report[2]} - TS_mF1:{report[6]}')
                if P <= 0:
                    break

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

def aggregate_loss(aspect_logits, polarity_logits, sample, parameters):
    attention_mask = sample['attention_mask'][:, 1:-1]
    aspect_logits = aspect_logits[:, 1:-1, :]
    polarity_logits = polarity_logits[:, 1:-1, :]
    mask = attention_mask.bool() 

    # Aspects loss
    flat_aspect_logits = aspect_logits.contiguous().view(-1, aspect_logits.shape[-1])
    flat_aspect_labels = sample['y_aspects'][:, 1:-1].contiguous().view(-1)
    selected_aspect_logits = flat_aspect_logits[mask.view(-1)]
    selected_aspect_labels = flat_aspect_labels[mask.view(-1)]
    aspect_loss = parameters['asp_criterion'](selected_aspect_logits, selected_aspect_labels)

    # Polarity loss
    flat_polarity_logits = polarity_logits.contiguous().view(-1, polarity_logits.shape[-1])
    flat_polarity_labels = sample['y_polarities'][:, 1:-1].contiguous().view(-1)
    selected_polarity_logits = flat_polarity_logits[mask.view(-1)]
    selected_polarity_labels = flat_polarity_labels[mask.view(-1)]
    polarity_loss = parameters['pol_criterion'](selected_polarity_logits, selected_polarity_labels)

    # Total loss
    loss = (parameters['aspect_loss_coeff'] * aspect_loss) + (parameters['polarity_loss_coeff'] * polarity_loss)

    if INFO_ENABLED:
        print(' - attention_mask:', attention_mask.shape, '\n',attention_mask)
        print(' - aspect_logits:', aspect_logits.shape, '\n',aspect_logits)
        print(' - polarity_logits:', polarity_logits.shape, '\n',polarity_logits)
        print(' - mask:', mask.shape, '\n',mask)
        print(' - flat_aspect_logits:', flat_aspect_logits.shape, '\n',flat_aspect_logits)
        print(' - flat_aspect_labels:', flat_aspect_labels.shape, '\n',flat_aspect_labels)
        print(' - selected_aspect_logits:', selected_aspect_logits.shape, '\n',selected_aspect_logits)
        print(' - selected_aspect_labels:', selected_aspect_labels.shape, '\n',selected_aspect_labels)
        print(' - aspect_loss:', aspect_loss)
        print(' - flat_polarity_logits:', flat_polarity_logits.shape, '\n',flat_polarity_logits)
        print(' - flat_polarity_labels:', flat_polarity_labels.shape, '\n',flat_polarity_labels)
        print(' - selected_polarity_logits:', selected_polarity_logits.shape, '\n',selected_polarity_logits)
        print(' - selected_polarity_labels:', selected_polarity_labels.shape, '\n',selected_polarity_labels)
        print(' - polarity_loss:', polarity_loss)
        print(' - loss:', loss)

    return loss

def extract_ote_ts(aspect_logits, polarity_logits, sample, parameters):

    attention_mask = sample['attention_mask'][:, 1:-1]
    aspect_logits = aspect_logits[:, 1:-1, :]
    polarity_logits = polarity_logits[:, 1:-1, :]
    aspect_mask = attention_mask.bool()

    pred_ot_list = []
    gold_ot_list = []

    # Decoding only aspects, for ot
    for i in range(aspect_logits.size(0)):

        pred_ot = torch.argmax(aspect_logits[i][aspect_mask[i]], dim=-1)
        gold_ot = sample['y_aspects'][i, 1:-1][aspect_mask[i]]
        pred_ot_list.append(parameters['lang'].decode_aspects(pred_ot.tolist()))
        gold_ot_list.append(parameters['lang'].decode_aspects(gold_ot.tolist()))

    # Polarity labels and aspects combined
    aspect_mask = attention_mask.bool()
    pred_ts_list = []
    gold_ts_list = []

    for i in range(polarity_logits.size(0)):

        selected_polarity_logits = torch.argmax(polarity_logits[i][aspect_mask[i]], dim=-1)
        gold_ts = sample['y_asppol'][i, 1:-1][aspect_mask[i]]
        pols = parameters['lang'].decode_polarities(selected_polarity_logits.tolist())
        gold_ts_decoded = parameters['lang'].decode_asppol(gold_ts.tolist())
        # Combining the predictions of aspects and related polarity
        pred_ts = [f'{ot}-{ts}' if ot != 'O' and ts != 'O' else 'O' for ot, ts in zip(pred_ot_list[i], pols)]
        pred_ts_list.append(pred_ts)
        gold_ts_list.append(gold_ts_decoded)

    if INFO_ENABLED:
        print('-gold_ot:', gold_ot_list)
        print('-pred_ot:', pred_ot_list)
        print('-gold_ts:', gold_ts_list)
        print('-pred_ts:', pred_ts_list)

    return gold_ot_list, gold_ts_list, pred_ot_list, pred_ts_list


def train_loop(data_loader, optimizer, model, parameters):
    model.train()
    losses = []
    for sample in data_loader:
        optimizer.zero_grad()
        aspect_logits, polarity_logits = model(sample['texts'], sample['attention_mask'], None)

        loss = aggregate_loss(aspect_logits, polarity_logits, sample, parameters)

        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
        optimizer.step()

    return losses

def eval_loop(data_loader, model, parameters):
    model.eval()
    gold_ot = []
    gold_ts = []
    pred_ot = []
    pred_ts = []
    losses = []

    with torch.no_grad():
        for sample in data_loader:
            aspect_logits, polarity_logits = model(sample['texts'], sample['attention_mask'], None)
            
            loss = aggregate_loss(aspect_logits, polarity_logits, sample, parameters)
            losses.append(loss.item())
            gold_ot_, gold_ts_, pred_ot_, pred_ts_ = extract_ote_ts(aspect_logits, polarity_logits, sample, parameters)

            gold_ot.extend(gold_ot_)
            gold_ts.extend(gold_ts_)
            pred_ot.extend(pred_ot_)
            pred_ts.extend(pred_ts_)

    ote_report, ts_report = evaluate(gold_ot, gold_ts, pred_ot, pred_ts)
    return losses, ote_report, ts_report

def plot_aligned_losses(training_losses, dev_losses, title):

    step = len(training_losses) / len(dev_losses)
    selected_training_losses = [training_losses[int(i * step)] for i in range(len(dev_losses))]

    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.plot(selected_training_losses, label='Training Loss')
    plt.plot(dev_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()