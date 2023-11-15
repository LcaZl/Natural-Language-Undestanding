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

    model = AspectSentimentModel(
        num_aspect_labels=parameters['output_aspects'],
        num_polarity_labels = parameters['output_polarities'],
        dropout_rate=parameters['dropout']
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

        train_loader, dev_loader, asp_weight, pol_weight = parameters['train_folds'][i]

        parameters['asp_criterion'] = nn.CrossEntropyLoss(weight = asp_weight, ignore_index = PAD_TOKEN)
        parameters['pol_criterion'] = nn.CrossEntropyLoss(weight = pol_weight, ignore_index = PAD_TOKEN)
        
        loss = train_loop(train_loader, optimizer, model, parameters)
        losses[loss_idx].append(loss)

        loss, ote_report, ts_report = eval_loop(dev_loader, model, parameters)
        losses[loss_idx].append(loss)

        f = round(ts_report[0], 3)
        recall = round(ts_report[2], 3)
        prec = round(ts_report[1], 3)
        fm = round(ts_report[3], 3)

        if f > best_score:
            best_score = f
            best_model = model
        
        pbar.set_description(f'FOLD {i+1} - F1:{f} - P:{prec} - R:{recall} - F1M:{fm}')

    loss, ote_report, ts_report = eval_loop(parameters['test_loader'], best_model, parameters)
    losses[loss_idx].append(loss)

    report = (model, [i, ote_report, ts_report])
    return best_model, report, losses


def aggregate_loss(aspect_logits, polarity_logits, sample, parameters):
    attention_mask = sample['attention_mask'][:, 1:-1]
    aspect_logits = aspect_logits[:, 1:-1, :] * attention_mask.unsqueeze(-1)
    polarity_logits = polarity_logits[:, 1:-1, :] * attention_mask.unsqueeze(-1)
    
    #print(' - attention_mask:', attention_mask.shape, '\n',attention_mask)
    #print(' - aspect_logits:', aspect_logits.shape, '\n',aspect_logits)
    #print(' - polarity_logits:', polarity_logits.shape, '\n',polarity_logits)
    # Maschera per gli aspetti identificati
    aspect_mask = (sample['y_aspects'][:, 1:-1] != 1) & (sample['y_aspects'][:, 1:-1] != PAD_TOKEN) & attention_mask.bool()
    #print(' - aspect_mask:', aspect_mask.shape, '\n',aspect_mask)

    # Calcolo della perdita per gli aspetti
    flat_aspect_logits = aspect_logits.contiguous().view(-1, aspect_logits.shape[-1])
    flat_aspect_labels = sample['y_aspects'][:, 1:-1].contiguous().view(-1)
    selected_aspect_logits = flat_aspect_logits[aspect_mask.view(-1)]
    selected_aspect_labels = flat_aspect_labels[aspect_mask.view(-1)]
    aspect_loss = parameters['asp_criterion'](selected_aspect_logits, selected_aspect_labels)
    #print(' - flat_aspect_logits:', flat_aspect_logits.shape, '\n',flat_aspect_logits)
    #print(' - flat_aspect_labels:', flat_aspect_labels.shape, '\n',flat_aspect_labels)
    #print(' - selected_aspect_logits:', selected_aspect_logits.shape, '\n',selected_aspect_logits)
    #print(' - selected_aspect_labels:', selected_aspect_labels.shape, '\n',selected_aspect_labels)
    #print(' - aspect_loss:', aspect_loss)

    # Calcolo della perdita per le polarità
    flat_polarity_logits = polarity_logits.contiguous().view(-1, polarity_logits.shape[-1])
    flat_polarity_labels = sample['y_polarities'][:, 1:-1].contiguous().view(-1)
    selected_polarity_logits = flat_polarity_logits[aspect_mask.view(-1)]
    selected_polarity_labels = flat_polarity_labels[aspect_mask.view(-1)]
    polarity_loss = parameters['pol_criterion'](selected_polarity_logits, selected_polarity_labels)
    #print(' - flat_polarity_logits:', flat_polarity_logits.shape, '\n',flat_polarity_logits)
    #print(' - flat_polarity_labels:', flat_polarity_labels.shape, '\n',flat_polarity_labels)
    #print(' - selected_polarity_logits:', selected_polarity_logits.shape, '\n',selected_polarity_logits)
    #print(' - selected_polarity_labels:', selected_polarity_labels.shape, '\n',selected_polarity_labels)
    #print(' - polarity_loss:', polarity_loss)

    loss = aspect_loss + polarity_loss
    #print(' - loss:', loss)

    return loss

def extract_ote_ts(aspect_logits, polarity_logits, sample, parameters):
    attention_mask = sample['attention_mask'][:,1:-1]
    aspect_logits = torch.argmax(aspect_logits[:,1:-1,:], dim=-1)
    polarity_logits = polarity_logits[:,1:-1,:]
    aspect_mask = (sample['y_aspects'][:, 1:-1] != 1) & (sample['y_aspects'][:, 1:-1] != PAD_TOKEN) & attention_mask.bool()
    print(' - attention_mask:', attention_mask.shape, '\n',attention_mask)
    print(' - aspect_logits:', aspect_logits.shape, '\n',aspect_logits)
    print(' - polarity_logits:', polarity_logits.shape, '\n',polarity_logits)
    print('- aspect_mask:', aspect_mask.shape, '\n', aspect_mask)

    flat_aspect_logits = aspect_logits.contiguous().view(-1, aspect_logits.shape[-1])
    flat_aspect_labels = sample['y_aspects'][:, 1:-1].contiguous().view(-1)
    print(' - flat_aspect_logits:', flat_aspect_logits.shape, '\n',flat_aspect_logits)
    print(' - flat_aspect_labels:', flat_aspect_labels.shape, '\n',flat_aspect_labels)
   
    pred_ot = flat_aspect_logits[aspect_mask.view(-1)]
    gold_ot = flat_aspect_labels[aspect_mask.view(-1)]

    print('- gold_ot:', gold_ot.shape, '', gold_ot)
    print('- pred_ot:', pred_ot.shape, '\n', pred_ot)

    pred_ot = [parameters['lang'].decode_aspects(el[mask]) for el, mask in pred_ot]
    gold_ot = [parameters['lang'].decode_aspects(el[mask]) for el, mask in gold_ot]

    print('- gold_ot:', len(gold_ot[0]), '', gold_ot[0])
    print('- pred_ot:', len(pred_ot[0]), '\n', pred_ot[0])

    exit(0)
    print('polarity_logit:', polarity_logits)
    print('aspect_logit', aspect_logits)
    gold_ot = sample['y_aspects'][:, 1:-1].contiguous()
    pred_ot = torch.argmax(aspect_logits, dim=-1)
    aspect_mask = (pred_ot != parameters['lang'].aspect2id['O']) & attention_mask.bool()
    aspect_mask = aspect_mask.bool().cpu().numpy()
    
    print('- gold_ot:', len(gold_ot[0]), '', gold_ot[0])
    print('- pred_ot:', len(pred_ot[0]), '\n', pred_ot[0])
    print('- aspect_mask:', len(aspect_mask[0]), '\n', aspect_mask[0])
    # Filtra i token di padding dalle etichette d'oro e dalle predizioni

    gold_ot = [parameters['lang'].decode_aspects(el[mask]) for el, mask in zip(gold_ot.cpu().numpy(), attention_mask.cpu().numpy())]
    pred_ot = [parameters['lang'].decode_aspects(el[mask]) for el, mask in zip(pred_ot.cpu().numpy(), attention_mask.cpu().numpy())]

    print('- gold_ot:', len(gold_ot[0]), '', gold_ot[0])
    print('- pred_ot:', len(pred_ot[0]), '\n', pred_ot[0])
    #print(evaluate_ote(gold_ot, pred_ot))

    gold_ts_2 = sample['y_asppol'][:, 1:-1].contiguous()
    pred_ts_2 = torch.argmax(polarity_logits, dim=-1)
    print('gold_ts_2',len(gold_ts_2[0]),'',gold_ts_2[0])
    print('pred_ts_2',len(pred_ts_2[0]),'',pred_ts_2[0])

    gold_ts_1 = [parameters['lang'].decode_asppol(el[mask]) for el, mask in zip(gold_ts_2.cpu().numpy(), attention_mask.cpu().numpy())]
    pred_ts_1 = [parameters['lang'].decode_polarities(el[mask]) for el, mask in zip(pred_ts_2.cpu().numpy(), attention_mask.cpu().numpy())]
    
    print('gold_ts_1',len(gold_ts_1[0]),'',gold_ts_1[0])
    print('pred_ts_1',len(pred_ts_1[0]),'',pred_ts_1[0])
    
    pred_ts = []
    gold_ts = []
    for idx in range(0, len(pred_ot)):
        tmp_p = []
        tmp_g = []
        for ot, ts, g_ts, is_aspect in zip(pred_ot[idx], pred_ts_1[idx], gold_ts_1[idx], aspect_mask[idx]):
            if is_aspect:
                label = f'{ot}-{ts}'
                tmp_p.append(label)
            else:
                tmp_p.append('O')
            tmp_g.append(g_ts)
        pred_ts.append(tmp_p)
        gold_ts.append(tmp_g)

    print('pred_ot',len(pred_ot[0]),'',pred_ot[0])
    print('-mask   :',aspect_mask[0])
    print('-gold_ot:',gold_ot[0])
    print('-pred_ot:',pred_ot[0])
    print('-gold_ts:',gold_ts[0])
    print('-pred_ts:',pred_ts[0])
    print(evaluate_ote(gold_ot, pred_ot))
    print(evaluate_ts(gold_ts, pred_ts))
    exit(0)

    return gold_ot, gold_ts, pred_ot, pred_ts

def train_loop(data_loader, optimizer, model, parameters):
    model.train()
    losses = []
    P = 3
    best_loss = math.inf
    for sample in data_loader:
        optimizer.zero_grad()
        aspect_logits, polarity_logits = model(sample['texts'], sample['attention_mask'], None)

        loss = aggregate_loss(aspect_logits, polarity_logits, sample, parameters)

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            P = 3
        else:
            P -= 1
            if P == 0:
                break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
        optimizer.step()

    return round(np.mean(losses), 3)

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

            print(evaluate_ote(gold_ot_, pred_ot_))
            print(evaluate_ts(gold_ts_, pred_ts_))

    ote_report, ts_report = evaluate(gold_ot, gold_ts, pred_ot, pred_ts)
    print(ote_report, ts_report)
    return round(np.mean(losses), 3), ote_report, ts_report
    