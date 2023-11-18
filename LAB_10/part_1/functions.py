# Add the class of your model only
# Here is where you define the architecture of your model using pytorch


import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import torch.optim as optim
from conll import evaluate
from sklearn.metrics import classification_report
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
from tabulate import tabulate
from utils import *
from model import *

def init_model(parameters, model_state = None):
    optimizer = None
    model = ModelIAS(hid_size = parameters['hidden_layer_size'], 
                                emb_size = parameters['embedded_layer_size'], 
                                out_slot = parameters['output_slots'], 
                                out_int = parameters['output_intent'], 
                                vocab_len = parameters['vocabulary_size'], 
                                pad_index = PAD_TOKEN).to(DEVICE)

    if model_state:
        model.load_state_dict(model_state)
    else:
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr = parameters['learning_rate'])

    return model, optimizer

def execute_experiment(exp_id, parameters):

    # Current experiment information
    print(f'\n-------- {exp_id} --------\n')
    print('Parameters:\n')
    for key, value in parameters.items():
        print(f' - {key}: {value}')

    model_filename = f"models_weight/IAS_{exp_id}.pth"

    if os.path.exists(model_filename):
        saved_data = torch.load(model_filename)
        print(f'Model founded. Parameters:', saved_data['parameters'])
        model, _ = init_model(parameters, saved_data['model_state'])

        best_model = (model, saved_data['best_report'])
        reports = saved_data['reports']
        losses = saved_data['losses']
        print(f' - Pre-trained model loaded.')

    else:
        parameters['criterion_slots'] = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN)
        parameters['criterion_intents'] = nn.CrossEntropyLoss() 

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

    pbar = range(parameters['runs'])
    for run in pbar:

        score, report = None, None
        model, optimizer = init_model(parameters)
        loss_idx = f'Run_{run}'
        train_losses[loss_idx], dev_losses[loss_idx] = [], []

        P = 3
        S = 0

        for epoch in tqdm(range(0,parameters['epochs'])):

            losses = train_loop(parameters, optimizer, model)
            train_losses[loss_idx].expand(losses)

            if epoch % 5 == 0:
                report_slot, report_intent, losses = eval_loop(parameters['dev_loader'],parameters, model)

                dev_losses[loss_idx].expand(losses)
                
                score = report_slot['total']['f']

                if score > S:
                    S = score
                    P = 3
                else:
                    P -= 1

                if P <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean
        
                pbar.set_description(f'Run {run} - Epoch {epoch} - L: {round(np.mean(losses), 3)} - S:{score} - Report:{report}')

        results_test, intent_test, _ = eval_loop(parameters['test_loader'], parameters, model)
        report = (run, results_test, intent_test)
        reports.append(report)
        score = results_test['total']['f']

        if score > best_score:
            best_score = score
            best_model = (model, report)    
            
    return best_model, reports, losses

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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
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
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return report_slot, report_intent, losses
