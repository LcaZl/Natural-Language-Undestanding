
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
import math

from utils import *
from model import *


def init_model(parameters, model_state = None):

    model = jointBERT(out_slot=parameters['num_slot_labels'],
                    out_int=parameters['num_intent_labels'],
                    dropout_rate=parameters['dropout_probabilities']).to(DEVICE)

    if model_state:
        model.load_state_dict(model_state)
    
    optimizer = optim.Adam(model.parameters(), lr = parameters['learning_rate'])

    return model, optimizer

def execute_experiment(exp_id, parameters):
        
    # Current experiment information
    print(f'\n-------- {exp_id} --------\n')
    print('Parameters:\n')
    for key, value in parameters.items():
        print(f' - {key}: {value}')

    print(f'\nStart Training:')

    model_filename = f"models/JointBert_{exp_id}.pth"

    if os.path.exists(model_filename):
        saved_data = torch.load(model_filename)
        model, _ = init_model(parameters, model_state=saved_data['model_state'])

        reports = saved_data['reports']
        best_model = (model, saved_data['best_report'])
        losses = saved_data['losses']
        parameters = saved_data['parameters']

        print(f' - Pre-trained model loaded.')

    else:

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

    return reports, best_model, losses

def train_lm(parameters):
    train_losses = {}
    dev_losses = {}
    best_score = 0
    reports = []

    pbar = tqdm(range(0,parameters['runs']))
    for r in pbar:

        score, report = None, None
        model, optimizer = init_model(parameters)
        loss_idx = f'run_{r}'
        train_losses[loss_idx] = []
        dev_losses[loss_idx] = []

        P = 3
        S = 0

        for epoch in range(0,parameters['epochs']):
            losses = train_loop(parameters['train_loader'], optimizer, model, parameters)
            train_losses[loss_idx].extend(losses)

            if epoch % 5 == 0:

                report_slot, report_intent, losses = eval_loop(parameters['dev_loader'], model, parameters)
                dev_losses[loss_idx].extend(losses)
                score = (report_slot['total']['f'] + report_intent['accuracy']) / 2

                if score > S:
                    S = score
                else:
                    P -= 1

                if P <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

                pbar.set_description(f'Run {r} - Epoch {epoch} - L: {round(np.mean(losses), 3)} - S:{score} - Report:{report}')

        report_slot, report_intent, _ = eval_loop(parameters['test_loader'], model, parameters)
        report = (r, report_slot, report_intent)
        reports.append(report)

        score = (report_slot['total']['f'] + report_intent['accuracy']) / 2

        if score > best_score:
            best_score = score
            best_model = (model, report)

    return best_model, reports, (train_losses, dev_losses)

def train_loop(data, optimizer, model, parameters):
               
    model.train()
    losses = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        intent_logits, slot_logits = model(sample['utterances'], sample['attention_mask'], sample['token_type_ids'])

        #print('- Sample & forward output:')
        #print("-    Intents   :", sample['intents'].shape)      
        #print("-   Utterances :", sample['utterances'].shape)
        #print("-   Att. Mask  :", sample['attention_mask'].shape)
        #print("-    Y Slots   :", sample['y_slots'].shape)
        #print("-   Slots Len  :", sample['slots_len'].shape)
        #print("- Intent logits:", intent_logits.shape)
        #print('-  Slot logits :', slot_logits.shape)

        # 1. Intent Loss
        intent_loss = parameters['criterion_intents'](intent_logits.view(-1, parameters['num_intent_labels']), sample['intents'].view(-1))
        
        # 2. Slot Loss
        if (parameters['use_crf']):
            slot_loss = -1 * model.crf(slot_logits, sample['y_slots'], mask=sample['attention_mask'].bool(), reduction='mean')
        else:
            active_loss = sample['attention_mask'].view(-1) == 1
            active_logits = slot_logits.view(-1, parameters['num_slot_labels'])[active_loss]
            active_labels = sample['y_slots'].view(-1)[active_loss]
            slot_loss = parameters['criterion_slots'](active_logits, active_labels)

        # 3. Total Loss
        total_loss = intent_loss + (slot_loss * parameters['slot_loss_coefficient'])
        losses.append(total_loss.item())

        total_loss.backward()

        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
        optimizer.step() # Update the weights

    return losses

def eval_loop(data, model, parameters):

    model.eval()

    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []
    losses = []

    with torch.no_grad():

        for sample in data:
            current_ref_slots, current_hyp_slots = [], []

            intent_logits, slot_logits = model(sample['utterances'], sample['attention_mask'], sample['token_type_ids'])

            # 1. Intent Loss
            intent_loss = parameters['criterion_intents'](intent_logits.view(-1, parameters['num_intent_labels']), sample['intents'].view(-1))

            # 2. Slot Loss
            if (parameters['use_crf']):
                slot_loss = -1 * model.crf(slot_logits, sample['y_slots'], mask=sample['attention_mask'].bool(), reduction='mean')
            else:
                active_loss = sample['attention_mask'].view(-1) == 1
                active_logits = slot_logits.view(-1, parameters['num_slot_labels'])[active_loss]
                active_labels = sample['y_slots'].view(-1)[active_loss]
                slot_loss = parameters['criterion_slots'](active_logits, active_labels)      

            # 3. Total Loss
            total_loss = intent_loss + (slot_loss * parameters['slot_loss_coefficient'])
            losses.append(total_loss.item())

            # Intent inference
            out_intents = [parameters['lang'].id2intent.get(x, '[UNK]') for x in torch.argmax(intent_logits, dim=1).tolist()]
            gt_intents = [parameters['lang'].id2intent.get(x, '[UNK]') for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            if (parameters['use_crf']):
                output_slots = model.decode_slots(slot_logits, sample['attention_mask'])
            else:
                output_slots = torch.argmax(slot_logits, dim=2).tolist()

            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq] - 1
                utt_ids = sample['utterance'][id_seq][1:length].tolist()
                gt_ids = sample['y_slots'][id_seq][1:length].tolist()
                gt_slots = [parameters['lang'].id2slot.get(elem, UNK_TOKEN) for elem in gt_ids]
                utterance = parameters['lang'].tokenizer.convert_ids_to_tokens(utt_ids)     

                to_decode = seq[1:length]
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                current_ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])

                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], parameters['lang'].id2slot[elem]))
                hyp_slots.append(tmp_seq)
                current_hyp_slots.append(tmp_seq)
    try:
        
        report_slot = evaluate(ref_slots, hyp_slots)

    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        report_slot = {'accuracy':0, 'total': {'f':0}}
        print('Class not in ref:', ref_s.difference(hyp_s),  hyp_s.difference(ref_s))
        print(' - utt_ids',utt_ids)
        print(' - gt_ids',gt_ids)
        print(' - gt_slots',gt_slots)
        print(' - utterance',utterance)
        print(' - to_decode',to_decode)
        print(' - ref_s',ref_s)
        print(' - hyp_s',hyp_s)
        print(ex)
        exit(0)
        
    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    
    return report_slot, report_intent, losses
