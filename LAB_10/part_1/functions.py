# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import numpy as np
import os
import torch.optim as optim
from conll import evaluate
from sklearn.metrics import classification_report
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

import json
from pprint import pprint
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD_TOKEN = 0
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
DEVICE = 'cuda:0'

from utils import *
from model import *

def execute_experiment(exp_id, parameters):

    # Current experiment information
    print(f'\n-------- {exp_id} --------\n')
    print('Parameters:\n')
    for key, value in parameters.items():
        print(f' - {key}: {value}')

    model_filename = f"models/IAS_{exp_id}.pth"

    if os.path.exists(model_filename):
        saved_data = torch.load(model_filename)
        print(f'Best model founded.\n\n Parameters:', saved_data['parameters'])
        model, _ = init_model(parameters, saved_data['model_state'])

        best_model = (model, saved_data['best_report'])
        reports = saved_data['reports']
        losses = saved_data['losses']
        print(f'\nPre-trained model loaded.')

    else:
        parameters['criterion_slots'] = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN)
        parameters['criterion_intents'] = nn.CrossEntropyLoss() 
        print('\nStart training:\n')
        best_model, reports, losses = train_lm(parameters)

        # Save model and scores
        data_to_save = {
            'model_state': best_model[0].state_dict(),
            'reports': reports,
            'best_report':best_model[1],
            'parameters':parameters,
            'losses':losses
        }
        torch.save(data_to_save, model_filename)

    return best_model, reports, losses

def train_lm(parameters):
    train_losses = {}
    dev_losses = {}
    best_score = 0
    reports = []

    pbar = tqdm(range(parameters['runs']))
    for run in pbar:

        score, report = None, None
        model, optimizer = init_model(parameters)
        loss_idx = f'Run_{run}'
        train_losses[loss_idx], dev_losses[loss_idx] = [], []

        P = 3
        S = 0

        for epoch in range(0,parameters['epochs']):

            tr_loss = train_loop(parameters, optimizer, model)

            if epoch % 2 == 0:
                report_slot, report_intent, dev_loss = eval_loop(parameters['dev_loader'],parameters, model)
                report = (report_slot, report_intent)
                train_losses[loss_idx].append(np.mean(tr_loss))
                dev_losses[loss_idx].append(np.mean(dev_loss))
                
                score = report_slot['total']['f']

                if score > S:
                    S = score
                else:
                    P -= 1

                if P <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean
        
                pbar.set_description(f'Run {run} - Epoch {epoch} - TL: {round(np.mean(tr_loss), 3)} - DL: {round(np.mean(dev_loss), 3)} - F1:{np.round(score,3)}')

        report_slot, report_intent, _ = eval_loop(parameters['test_loader'], parameters, model)
        report = (run, report_slot, report_intent)
        reports.append(report)
        score = report_slot['total']['f']

        if score > best_score:
            best_score = score
            best_model = (model, report)    
            
    return best_model, reports, (train_losses, dev_losses)


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
    optimizer = None
    model = ModelIAS(hid_size = parameters['hidden_layer_size'], 
                                emb_size = parameters['embedded_layer_size'], 
                                out_slot = parameters['output_slots'], 
                                out_int = parameters['output_intent'], 
                                vocab_len = parameters['vocabulary_size']).to(DEVICE)

    if model_state:
        model.load_state_dict(model_state)
    else:
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr = parameters['learning_rate'])

    return model, optimizer

def train_loop(parameters, optimizer, model):
    model.train()
    losses = []
    for sample in parameters['train_loader']:
        optimizer.zero_grad() # Zeroing the gradient
        
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = parameters['criterion_intents'](intent, sample['intents'])
        loss_slot = parameters['criterion_slots'](slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses.
                                       # Is there another way to do that?

        losses.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
        optimizer.step() # Update the weights
    return losses

def eval_loop(data, parameters, model):
    model.eval()
    losses = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = parameters['criterion_intents'](intents, sample['intents'])
            loss_slot = parameters['criterion_slots'](slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            losses.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [parameters['lang'].id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [parameters['lang'].id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [parameters['lang'].id2slot[elem] for elem in gt_ids[:length]]
                utterance = [parameters['lang'].id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], parameters['lang'].id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        report_slot = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print('Exception:', ex)
        report_slot = None

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return report_slot, report_intent, losses

# For outputs
def get_scores(reports, experiment_id):
    df = pd.DataFrame(columns=['Experiment ID','Intent accuracy', 'Accuracy std.', 'Slot F1 score', 'F1 std.'])

    slot_f1, intent_acc = [], []

    for [run, slot_report, intent_report] in reports:
        slot_f1.append(slot_report['total']['f'])
        intent_acc.append(intent_report['accuracy'])
        df.loc[len(df)] = [f'{experiment_id}_run_{run + 1}', slot_report['total']['f'], 0, intent_report['accuracy'], 0]
    df.loc[len(df)] = [f'{experiment_id}_avg', np.mean(slot_f1), np.std(slot_f1), np.mean(intent_acc), np.std(intent_acc)]
    df = df.round(4)

    return df

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