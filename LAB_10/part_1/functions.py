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

def init_model(parameters):

def execute_experiments(experiments_parameters):

    cols = ['Id','Run','Accuracy','Accuracy Std','F score', 'F Std']
    scores = pd.DataFrame(columns = cols)

    for exp_id, parameters in experiments_parameters.items():
        # Current experiment information
        print(f'\n-------- {exp_id} --------\n')
        print('Parameters:\n')
        for key, value in parameters.items():
            print(f' - {key}: {value}')

        print(f'\nStart Training:')
        slot_f1s, intent_acc = [], []

        print(f'- Run {run}')
        model_filename = f"models_weight/{exp_id}_run{run}.pth"

        if os.path.exists(model_filename):
            saved_data = torch.load(model_filename)
            print(f'Model founded. Parameters:', saved_data['parameters'])

            model = ModelIAS(hid_size = parameters['hidden_layer_size'], 
                                emb_size = parameters['embedded_layer_size'], 
                                out_slot = parameters['output_slots'], 
                                out_int = parameters['output_intent'], 
                                vocab_len = parameters['vocabulary_size'], 
                                pad_index = PAD_TOKEN).to(DEVICE)
            model.load_state_dict(saved_data['model_state'])

            f1 = saved_data['slot_f1']
            accuracy = saved_data['intent_acc']
            intent_acc.append(accuracy)
            slot_f1s.append(f1)
            print(f' - Pre-trained model loaded.')

        else:
            if parameters['model'] == 'IAS':
                model = ModelIAS(hid_size = parameters['hidden_layer_size'], 
                                    emb_size = parameters['embedded_layer_size'], 
                                    out_slot = parameters['output_slots'], 
                                    out_int = parameters['output_intent'], 
                                    vocab_len = parameters['vocabulary_size'], 
                                    pad_index = PAD_TOKEN).to(DEVICE)
                model.apply(init_weights)
            
            optimizer = optim.Adam(model.parameters(), lr = parameters['learning_rate'])
            criterion_slots = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN)
            criterion_intents = nn.CrossEntropyLoss() 
            
            accuracy, f1 = train_lm(model, parameters, optimizer, criterion_slots, criterion_intents)
            intent_acc.append(accuracy)
            slot_f1s.append(f1)

            # Save model and scores
            data_to_save = {
                'model_state': model.state_dict(),
                'slot_f1': f1,
                'intent_acc': accuracy,
                'parameters':parameters
            }
            torch.save(data_to_save, model_filename)

        experiment_result = pd.DataFrame(columns=cols, 
                                data = [[exp_id, run, accuracy, 0, f1, 0]])
        print(tabulate(experiment_result, headers='keys', tablefmt='grid', showindex=True))
        scores = pd.concat([scores, experiment_result])

        slot_f1s = np.asarray(slot_f1s)
        intent_acc = np.asarray(intent_acc)
        f1_avg = round(slot_f1s.mean(),3)
        f1_std = round(slot_f1s.std(),3)
        accuracy_avg = round(intent_acc.mean(), 3)
        accuracy_std = round(intent_acc.std(), 3)
        experiment_result = pd.DataFrame(columns=cols, 
                                data = [[exp_id, 'Average', accuracy_avg, accuracy_std, f1_avg, f1_std]])
        
        scores = pd.concat([scores, experiment_result])
        #print(tabulate(scores, headers='keys', tablefmt='grid', showindex=True))

    return scores

def train_lm(model, parameters, optimizer, criterion_slots, criterion_intents):
    losses = {}
    best_f1 = 0
    pbar = range(parameters['runs'])
    for run in pbar:
        score, report = None, None
        model, optimizer = init_model(parameters)
        loss_idx = f'Run_{i}'
        losses[loss_idx] = []
        P = 3
        S = 0

        for x in tqdm(range(0,parameters['epochs'])):

            loss = train_loop(parameters['train_loader'], optimizer, criterion_slots, criterion_intents, model)
            losses[loss_idx].append(loss)

            if x % 5 == 0:
                results_dev, intent_dev, loss_dev = eval_loop(parameters['dev_loader'], 
                                                            criterion_slots,
                                                            criterion_intents, model, 
                                                            parameters['lang'])
                
                losses[loss_idx].append(loss)
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                    patience = 3
                else:
                    patience -= 1

                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(parameters['test_loader'], criterion_slots,
                                            criterion_intents, model, parameters['lang'])

        if
    
    return intent_test['accuracy'], results_test['total']['f']

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



def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses.
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array
