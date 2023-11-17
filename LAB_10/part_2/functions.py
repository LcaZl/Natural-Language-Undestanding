
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

def execute_experiments(experiments_parameters):

    cols = ['Id','Run','Accuracy','Accuracy Std.','F-score', 'F-score Std.']
    scores = pd.DataFrame(columns = cols)

    for exp_id, parameters in experiments_parameters.items():
        
        # Current experiment information
        print(f'\n-------- {exp_id} --------\n')
        print('Parameters:\n')
        for key, value in parameters.items():
            print(f' - {key}: {value}')

        print(f'\nStart Training:')
        slot_f1s, intent_acc, slsf_accuracies = [], [], []

        for run in range(parameters['runs']):
            print(f'- Run {run}')
            model_filename = f"models_weight/{exp_id}_run{run}.pth"

            if os.path.exists(model_filename):
                saved_data = torch.load(model_filename)
                
                model = jointBERT(out_slot=parameters['num_slot_labels'],
                                 out_int=parameters['num_intent_labels'],
                                 dropout_rate=parameters['dropout_probabilities']).to(DEVICE)
                model.load_state_dict(saved_data['model_state'])

                f1 = saved_data['slot_f1']
                accuracy = saved_data['intent_acc']
                train_losses = saved_data['train_losses']
                eval_losses = saved_data['eval_losses']

                intent_acc.append(accuracy)
                slot_f1s.append(f1)
                print(f' - Pre-trained model loaded.')

            else:
                model = jointBERT(out_slot=parameters['num_slot_labels'],
                                 out_int=parameters['num_intent_labels'],
                                 dropout_rate=parameters['dropout_probabilities']).to(DEVICE)
                
                optimizer = optim.Adam(model.parameters(), lr = parameters['learning_rate'])
                report_slot, report_intent, train_losses, eval_losses = train_lm(model, parameters, optimizer)

                f1 = report_slot['total']['f']
                accuracy = report_intent['accuracy']
                intent_acc.append(accuracy)
                slot_f1s.append(f1)

                # Save model and scores
                data_to_save = {
                    'model_state': model.state_dict(),
                    'slot_f1': f1,
                    'intent_acc': accuracy,
                    'train_losses':train_losses,
                    'eval_losses':eval_losses
                }
                torch.save(data_to_save, model_filename)

            data = [exp_id, run, accuracy, 0, f1, 0]
            experiment_result = pd.DataFrame(columns=cols, data = [data])
            scores = pd.concat([scores, experiment_result])
            print(tabulate(experiment_result, headers='keys', tablefmt='grid', showindex=True))
            print('-  Training losses :', train_losses)
            print('- Evaluation losses:', eval_losses)

        slot_f1s = np.asarray(slot_f1s)
        intent_acc = np.asarray(intent_acc)
        f1_avg = round(slot_f1s.mean(),3)
        f1_std = round(slot_f1s.std(),3)
        accuracy_avg = round(intent_acc.mean(), 3)
        accuracy_std = round(intent_acc.std(), 3)

        experiment_result = pd.DataFrame(columns=cols, 
                                data = [[exp_id, 'Average', accuracy_avg, accuracy_std, f1_avg, f1_std]])
        
        scores = pd.concat([scores, experiment_result])
        print(tabulate(scores, headers='keys', tablefmt='grid', showindex=True))

    return scores

def train_lm(model, parameters, optimizer):
    losses_train = []
    losses_eval = []
    sampled_epochs = []
    best_loss = math.inf
    patience = 3
    for x in tqdm(range(0,parameters['epochs'])):
        loss, _, _ = train_loop(parameters['train_loader'], optimizer, model, parameters)
        losses_train.append(np.asarray(loss).mean())

        if x % 5 == 0:
            sampled_epochs.append(x)

            _, _, eval_total_loss = eval_loop(parameters['dev_loader'], model, parameters)
            eval_mean_loss = np.asarray(eval_total_loss).mean()
            losses_eval.append(eval_mean_loss)

            if eval_mean_loss < best_loss:
                best_loss = eval_mean_loss
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    report_slot, report_intent, eval_total_loss = eval_loop(parameters['test_loader'], model, parameters)
    eval_mean_loss = np.asarray(eval_total_loss).mean()
    losses_eval.append(eval_mean_loss)

    #plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    #plt.title('Train and Dev Losses')
    #plt.ylabel('Loss')
    #plt.xlabel('Epochs')
    #plt.plot(sampled_epochs, losses_train, label='Train loss')
    #plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    #plt.legend()
    #plt.show()

    return report_slot, report_intent, losses_train, losses_eval

def train_loop(data, optimizer, model, parameters):
               
    model.train()
    total_loss_array = []
    intent_loss_array = []
    slot_loss_array = []

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
        intent_loss_array.append(intent_loss.item())
        
        # 2. Slot Loss
        if (parameters['use_crf']):
            slot_loss = -1 * model.crf(slot_logits, sample['y_slots'], mask=sample['attention_mask'].bool(), reduction='mean')
        else:
            active_loss = sample['attention_mask'].view(-1) == 1
            active_logits = slot_logits.view(-1, parameters['num_slot_labels'])[active_loss]
            active_labels = sample['y_slots'].view(-1)[active_loss]
            slot_loss = parameters['criterion_slots'](active_logits, active_labels)

        slot_loss_array.append(slot_loss.item())        

        # 3. Total Loss
        total_loss = intent_loss + (slot_loss * parameters['slot_loss_coefficient'])
        total_loss_array.append(total_loss.item())

        # Compute the gradient, deleting the computational graph    
        total_loss.backward()

        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])
        optimizer.step() # Update the weights

    return total_loss_array, intent_loss_array, slot_loss_array

def eval_loop(data, model, parameters):

    model.eval()

    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []
    intent_loss_array, slot_loss_array, total_loss_array = [], [], []
    with torch.no_grad():  # Avoid the creation of a computational graph

        for sample in data:
            current_ref_slots, current_hyp_slots = [], []

            intent_logits, slot_logits = model(sample['utterances'], sample['attention_mask'], sample['token_type_ids'])

            # 1. Intent Loss
            intent_loss = parameters['criterion_intents'](intent_logits.view(-1, parameters['num_intent_labels']), sample['intents'].view(-1))
            intent_loss_array.append(intent_loss.item())

            # 2. Slot Loss
            if (parameters['use_crf']):
                slot_loss = -1 * model.crf(slot_logits, sample['y_slots'], mask=sample['attention_mask'].bool(), reduction='mean')
            else:
                active_loss = sample['attention_mask'].view(-1) == 1
                active_logits = slot_logits.view(-1, parameters['num_slot_labels'])[active_loss]
                active_labels = sample['y_slots'].view(-1)[active_loss]
                slot_loss = parameters['criterion_slots'](active_logits, active_labels)      
            slot_loss_array.append(slot_loss.item())

            # 3. Total Loss
            total_loss = intent_loss + (slot_loss * parameters['slot_loss_coefficient'])
            total_loss_array.append(total_loss.item())

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
    
    return report_slot, report_intent, total_loss_array
