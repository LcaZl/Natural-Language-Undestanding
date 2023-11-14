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
        num_polarity_labels = parameters['output_polarities']
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
        ote_score, ts_score = report

        f = round(ts_score[0], 3)
        recall = round(ts_score[2], 3)
        prec = round(ts_score[1], 3)
        fm = round(ts_score[3], 3)

        if f > best_score:
            best_score = f
            best_model = model
        
        pbar.set_description(f'FOLD {i+1} - F1:{f} - P:{prec} - R:{recall} - F1M:{fm}')

    loss, report = eval_loop(parameters['test_loader'], best_model, parameters)
    losses[loss_idx].append(loss)
    ote_score, ts_score = report

    report = (model, [i, ote_score, ts_score])
    return best_model, report, losses

def train_loop(data_loader, optimizer, model, parameters):
    model.train()
    losses = []

    for sample in data_loader:

        optimizer.zero_grad()
        aspect_logits, polarity_logits = model(sample['texts'], sample['attention_mask'], None)
        
        aspect_preds = torch.argmax(aspect_logits, dim=-1)
        polarity_preds = torch.argmax(active_polarity_logits, dim=-1)

        aspect_mask = aspect_preds != 0

        active_polarity_logits = polarity_logits.view(-1, polarity_logits.shape[-1])[aspect_mask.view(-1)]
        active_polarity_labels = sample['y_polarities'].view(-1)[aspect_mask.view(-1)]
        
        active_aspect_logits = aspect_logits.view(-1, aspect_logits.shape[-1])
        aspect_loss = parameters['criterion'](active_aspect_logits, sample['y_aspects'].view(-1))
        polarity_loss = parameters['criterion'](active_polarity_logits, active_polarity_labels)
        loss = aspect_loss + polarity_loss  # Considera di pesare diversamente le due loss se necessario

        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
        optimizer.step()

    return round(np.mean(losses), 3)


def eval_loop(data_loader, model, parameters):
    model.eval()
    pred_ot = []
    gold_ot = []
    polarity_preds = []
    gold_ts = []
    losses = []
    aspect_masks = []

    with torch.no_grad():
        for sample in data_loader:
            aspect_logits, polarity_logits = model(sample['texts'], sample['attention_mask'], sample['token_type_ids'])
            
            # More probable predicted classes of aspect and thus the identified aspects mask
            aspect_preds = torch.argmax(aspect_logits, dim=-1)
            aspect_mask = aspect_preds != 0

            # Use only polarity of identified aspects
            active_polarity_logits = polarity_logits.view(-1, polarity_logits.shape[-1])[aspect_mask.view(-1)]

            # Use only polarity of indentified aspects from ground truth
            active_polarity_labels = sample['y_polarities'].view(-1)[aspect_mask.view(-1)]
            
            # Compute loss
            aspect_loss = parameters['criterion'](aspect_logits.view(-1, aspect_logits.shape[-1]), sample['y_aspects'].view(-1))
            polarity_loss = parameters['criterion'](active_polarity_logits, active_polarity_labels)
            loss = aspect_loss + polarity_loss  # Considera di pesare diversamente le due loss se necessario
            losses.append(loss.item())
            
            pred_aspect_tags = parameters['lang'].decode_aspects(aspect_preds.view(-1).cpu().numpy())
            active_polarity_preds = torch.argmax(active_polarity_logits, dim=-1)
            pred_polarity_tags = parameters['lang'].decode_polarities(active_polarity_preds.cpu().numpy())

            print(aspect_logits.shape, aspect_preds.shape, )
            print('asptag1', aspect_preds)
            print('asptag2:',pred_aspect_tags)
            print('poltags:',pred_polarity_tags)
            exit(0)

            active_mask = sample['attention_mask'].view(-1) == 1
            active_aspect_logits = aspect_logits.view(-1, aspect_logits.shape[-1])[active_mask]
            active_polarity_logits = polarity_logits.view(-1, polarity_logits.shape[-1])[active_mask & aspect_mask.view(-1)]
            
            # Ora calcola le probabilità e le predizioni solo per i token attivi
            aspect_probs = torch.softmax(active_aspect_logits, dim=1)
            pred_ot_batch = torch.argmax(aspect_probs, dim=1)
            pred_ot.append(pred_ot_batch.cpu().numpy())

            polarity_probs = torch.softmax(active_polarity_logits, dim=1)
            polarity_preds_batch = torch.argmax(polarity_probs, dim=1)
            polarity_preds.append(polarity_preds_batch.cpu().numpy())

            active_gold_ot = sample['y_aspects'].view(-1)[active_mask]
            gold_ot.append(active_gold_ot.cpu().numpy())

            gold_ts_labels = sample['y_asppol'].view(-1)[active_mask]
            gold_ts.append(gold_ts_labels.cpu().numpy())

            aspect_masks.append([el != 0 for el in pred_ot_batch.cpu().numpy()])
            
            if INFO_ENABLED:
                print('- pred_ot   :', parameters['lang'].decode_aspects(pred_ot_batch.cpu().numpy()))
                print('- gold_ot  :', parameters['lang'].decode_aspects(active_gold_ot.cpu().numpy()))
                print('- polarity_preds :', parameters['lang'].decode_polarities(polarity_preds_batch.cpu().numpy()))
                print('- gold_ts:', parameters['lang'].decode_asppol(gold_ts_labels))


    if not INFO_ENABLED:
        print(len(gold_ot),len(gold_ts),len(pred_ot))
        print('- gold_ot', gold_ot[0])
        print('- gold_ts', gold_ts[0])
        print('- pred_ot', pred_ot[0])
        print('- Aspects mask:', aspect_mask[0])

    polarity_preds = [parameters['lang'].decode_polarities(lbs) for lbs in polarity_preds]
    gold_ot = [parameters['lang'].decode_aspects(lbs) for lbs in gold_ot]
    gold_ts = [parameters['lang'].decode_asppol(lbs) for lbs in gold_ts]
    pred_ot = [parameters['lang'].decode_aspects(lbs) for lbs in pred_ot]

    pred_ts = []
    if (len(pred_ot) == len(polarity_preds)):
        for asps, pols, mask in zip(pred_ot, polarity_preds, aspect_masks):
            assert len(asps) == len(pols) == len(mask)
            preds = []
            for asp, pol, is_aspect in zip(asps, pols, mask):
                label = f'{asp}-{pol}' if is_aspect else 'O'
                preds.append(label)
            pred_ts.append(preds)

    if not INFO_ENABLED:
        print(len(gold_ot),len(gold_ts),len(pred_ot),len(pred_ts))
        print('- gold_ot', gold_ot[0])
        print('- gold_ts', gold_ts[0])
        print('- pred_ot', pred_ot[0])
        print('- pred_ts', pred_ts[0])
        print('- Aspects mask:', aspect_mask[0])

    report1 = evaluate_ote(gold_ot, pred_ot) # (PRECISION, RECALL, F1)
    print('evaluate report:', report1)
    report2 = evaluate_ts(gold_ts, pred_ts)
    print('evaluate report:', report2)

    return round(np.mean(losses), 3), (report1, report2)
    