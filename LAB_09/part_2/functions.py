# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import pandas as pd
from tabulate import tabulate
import math
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader

from utils import *
from model import *
def execute_experiments(experiments):
        """
        This function iterates over a dictionary of experiments. For each of them it trains a language model
        based on the specified parameters, evaluates the model, and logs the results.
        
        Parameters:
        - experiments (dict): dictionary of experiments with relative parameters
        
        Outputs:
        - scores (pd.DataFrame): dataFrame containing the perplexity of each model trained
        """
        cols = ['Experiment ID','Model','PPL']
        scores = pd.DataFrame(columns = cols)
        
        for experiment_id, experiment_parameters in experiments.items():

            # Current experiment information
            print(f'\n-------- {experiment_id} --------\n')
            print('Parameters:\n')
            for key, value in experiment_parameters.items():
                print(f' - {key}: {value}')

            sents_len = [experiment_parameters['train_max_len'], experiment_parameters['dev_max_len'], experiment_parameters['test_max_len']]
            # Experiment also with a smaller or bigger model by changing hid and emb sizes 
            model = LM_(
                        nn_type = experiment_parameters['model_name'],
                        emb_size = experiment_parameters['embedded_layer_size'], 
                        hidden_size = experiment_parameters['hidden_layer_size'], 
                        output_size = experiment_parameters['vocab_len'], 
                        pad_index = experiment_parameters['lang'].word2id["<pad>"],
                        out_dp_prob = experiment_parameters['output_dropout'],
                        emb_dp_prob = experiment_parameters['embedding_dropout'],
                        weight_tying = experiment_parameters['weight_tying'],
                        variational_dropout = experiment_parameters['variational_dropout'],
                        max_len = max(sents_len)).to(experiment_parameters['device'])
            
            model.apply(init_weights)

            # Defining the optimizer
            optimizer = None            
            if experiment_parameters['optmz_type'] == 'SGD' or experiment_parameters['optmz_type'] == 'NT-AvSGD':
                optimizer = optim.SGD(model.parameters(), 
                                    lr=experiment_parameters['optmz_learning_rate'])
            elif experiment_parameters['optmz_type'] == 'Adam':
                optimizer = optim.AdamW(model.parameters(), 
                                            lr=experiment_parameters['optmz_learning_rate'])
                
            # If weights of the model altready exists, don't train.
            model_score = load_model_and_print_ppl(model, 
                                                    experiment_parameters['weight_path'], 
                                                    experiment_parameters['test_loader'], 
                                                    experiment_parameters['criterion_eval'], 
                                                    experiment_parameters['device'])
            
            if model_score is None: # If the weight in specified folder doesn't exists
                print(f'\nStart training ... \n')

                if experiment_parameters['optmz_type'] == 'NT-AvSGD':
                    model_score = train_lm_nt_avsgd(
                            experiment_parameters['weight_path'],
                            experiment_parameters['n_epochs'], 
                            experiment_parameters['patience'], 
                            model, 
                            optimizer, 
                            experiment_parameters['criterion_train'], 
                            experiment_parameters['criterion_eval'], 
                            experiment_parameters['train_loader'], 
                            experiment_parameters['dev_loader'], 
                            experiment_parameters['test_loader'], 
                            experiment_parameters['device'], 
                            experiment_parameters['clip'],
                            experiment_parameters['logging_interval'],
                            experiment_parameters['non_monotonic_interval'])
                else:
                    model_score = train_lm(
                            experiment_parameters['weight_path'],
                            experiment_parameters['n_epochs'], 
                            experiment_parameters['patience'], 
                            model, 
                            optimizer, 
                            experiment_parameters['criterion_train'], 
                            experiment_parameters['criterion_eval'], 
                            experiment_parameters['train_loader'], 
                            experiment_parameters['dev_loader'], 
                            experiment_parameters['test_loader'], 
                            experiment_parameters['device'], 
                            experiment_parameters['clip'])
            
            experiment_result = pd.DataFrame(columns=cols, 
                                 data = [[experiment_id, experiment_parameters['model_name'], model_score]])
            
            print(f'{experiment_id} final score:\n')
            print(tabulate(experiment_result, headers='keys', tablefmt='grid', showindex=True), '\n')
            scores = pd.concat([scores, experiment_result])
            
        return scores

def train_lm(save_path, n_epochs, patience, model, optimizer, criterion_train, criterion_eval, train_loader, dev_loader, test_loader, device, 
             clip):
        """
        Train a language model

        Parameters:
        - save_path (str): path where the model with best performance is saved
        - n_epochs (int): maximum number of epochs to train the model
        - patience (int): number of epochs to wait for improvement before stopping training
        - model (torch.nn.Module): PyTorch model to train
        - optimizer (torch.optim.Optimizer): optimizer to use during training
        - criterion_train (torch.nn.Module): loss function used during training
        - criterion_eval (torch.nn.Module): loss function used during evaluation
        - train_loader (torch.utils.data.DataLoader): dataLoader for the training set
        - dev_loader (torch.utils.data.DataLoader): dataLoader for the development set
        - test_loader (torch.utils.data.DataLoader): dataLoader for the test set
        - device (str or torch.device): device, "cpu" or "cuda"
        - clip (float): value to clip gradients during training

        Returns:
        - final_ppl (float): perplexity of the best model on training set
        """  
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1,n_epochs))

        for epoch in pbar:
            if model.variational_dropout:
                model.reset_vd_mask()

            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            
            if epoch % 1 == 0:
                
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                pbar.set_description("PPL: %f" % ppl_dev)

                if  ppl_dev < best_ppl:  

                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    P = patience

                else:
                    P -= 1
                    
                if P <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean 

        best_model.to(device)
        torch.save(best_model.state_dict(), save_path)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)            
        return final_ppl

def average_weights(weight_list):
    """
    Return the average of the weights inside the input list
    """
    avg_weights = {}
    for key in weight_list[0].keys():
        avg_weights[key] = sum([weights[key] for weights in weight_list]) / len(weight_list)
    return avg_weights

def train_lm_nt_avsgd(save_path, n_epochs, patience, model, optimizer, criterion_train, criterion_eval, train_loader, dev_loader, test_loader, device, 
             clip, L, n):
        
        """
        Train a language model using non monotonic averaged SGD.

        Parameters:
        - save_path (str): path where the model with best performance is saved
        - n_epochs (int): maximum number of epochs to train the model
        - patience (int): number of epochs to wait for improvement before stopping training
        - model (torch.nn.Module): PyTorch model to train
        - optimizer (torch.optim.Optimizer): optimizer to use during training
        - criterion_train (torch.nn.Module): loss function used during training
        - criterion_eval (torch.nn.Module): loss function used during evaluation
        - train_loader (torch.utils.data.DataLoader): dataLoader for the training set
        - dev_loader (torch.utils.data.DataLoader): dataLoader for the development set
        - test_loader (torch.utils.data.DataLoader): dataLoader for the test set
        - device (str or torch.device): device, "cpu" or "cuda"
        - clip (float): value to clip gradients during training
        - L 

        Returns:
        - final_ppl (float): perplexity of the best model on training set
        """  
        # Initialize a variable to store the best perplexity (PPL) value
        logs = []
        saved_weights = []
        T = 0
        t = 0
        best_ppl = math.inf

        pbar = tqdm(range(n_epochs))  # Inizio da 0
        for epoch_K in pbar:

            if model.variational_dropout:
                model.reset_vd_mask()
                
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
                        
            if epoch_K % L == 0 and T == 0:  # Controlla se questa condizione è mai vera                      
                # Evaluate the model on the validation set
                PPL, _ = eval_loop(dev_loader, criterion_eval, model)
                pbar.set_description("PPL: %f" % PPL)

                # If the current PPL is better than the best PPL
                if t > n and PPL > min(logs[:t-n+1]): 
                    T = epoch_K

                logs.append(PPL)
                t = t + 1

                if PPL < best_ppl:
                    best_ppl = PPL
                    P = patience
                else:
                    P -= 1

                if patience <= 0:
                    break
            else:
                saved_weights.append(copy.deepcopy(model.state_dict()))

        # Average the saved weights to implement NT-AvSGD
        avg_weights = average_weights(saved_weights)   
        model.load_state_dict(avg_weights)
        model.to(device)
        torch.save(model.state_dict(), save_path)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)            
        return final_ppl

def load_model_and_print_ppl(model, model_path, test_loader, criterion_eval, device):
    """
    Load a model and print its Perplexity (PPL) on the test dataset.

    Parameters:
    - model (torch.nn.Module): untrained model architecture
    - model_path (str): path to the file containing the saved model parameters
    - test_loader (torch.utils.data.DataLoader): dataLoader for the test set
    - criterion_eval (torch.nn.Module): loss function used during evaluation
    - device (str or torch.device): device
    
    Returns:
    - ppl (float): perplexity of the model
    """
    
    # Check if the model file exists at the specified path.
    if not os.path.exists(model_path):
        print(f"\n-> No model file found at {model_path}.\n")
        return None
    print(f'\n-> Weights founded! Loading ...\n')

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    ppl, _ = eval_loop(test_loader, criterion_eval, model)
    
    return ppl

## from LAB_09
def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

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