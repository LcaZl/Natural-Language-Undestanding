# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import pandas as pd
from tabulate import tabulate
import math
import torch.optim as optim
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
            model = LangModel(
                        emb_size = experiment_parameters['embedded_layer_size'], 
                        hidden_size = experiment_parameters['hidden_layer_size'], 
                        output_size = experiment_parameters['vocab_len'], 
                        pad_index = experiment_parameters['lang'].word2id["<pad>"],
                        out_dp_prob = experiment_parameters['output_dropout'],
                        emb_dp_prob = experiment_parameters['embedding_dropout'],
                        weight_tying = experiment_parameters['weight_tying'],
                        variational_dropout = experiment_parameters['variational_dropout'],
                        n_layers=experiment_parameters['layers'],
                        max_len = max(sents_len)).to(DEVICE)
            
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
            model_score = load_model_and_print_ppl(model, experiment_parameters)
            
            if model_score is None: # If the weight in specified folder doesn't exists
                print(f'\nStart training ... \n')

                if experiment_parameters['optmz_type'] == 'NT-AvSGD':
                    model_score = train_LangModelnt_avsgd(experiment_parameters, model,  optimizer)
                else:
                    model_score = train_lm(experiment_parameters, model, optimizer)
            
            experiment_result = pd.DataFrame(columns=cols, 
                                 data = [[experiment_id, experiment_parameters['model_name'], model_score]])
            
            print(f'{experiment_id} final score:\n')
            print(tabulate(experiment_result, headers='keys', tablefmt='grid', showindex=True), '\n')
            scores = pd.concat([scores, experiment_result])
            
        return scores

def train_lm(parameters, model, optimizer):
        """
        Train a language model

        Parameters:
        - parameters (dict): parameters for the procedure
        - model (torch.nn.Module): PyTorch model to train
        - optimizer (torch.optim.Optimizer): optimizer to use during training


        Returns:
        - final_ppl (float): perplexity of the best model on training set
        """  
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(0,parameters['n_epochs']))

        for epoch in pbar:

            loss = train_loop(parameters['train_loader'], optimizer, parameters['criterion_train'], model, parameters['clip'])    
            
            if epoch % 1 == 0:
                
                ppl_dev, loss_dev = eval_loop(parameters['dev_loader'], parameters['criterion_eval'], model)
                pbar.set_description("PPL: %f" % ppl_dev)

                if  ppl_dev < best_ppl:  

                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    P = parameters['patience']
                else:
                    P -= 1
                    
                if P <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean 

        best_model.to(DEVICE)
        torch.save(best_model.state_dict(), parameters['weight_path'])
        final_ppl,  _ = eval_loop(parameters['test_loader'], parameters['criterion_eval'], best_model)            
        return final_ppl

def average_weights(weight_list):
    """
    Return the average of the weights inside the input list
    """
    avg_weights = {}
    for key in weight_list[0].keys():
        avg_weights[key] = sum([weights[key] for weights in weight_list]) / len(weight_list)
    return avg_weights

def train_LangModelnt_avsgd(parameters, model, optimizer):
        """
        Train a language model using non monotonic averaged SGD.

        Parameters:
        - parameters (dict): parameters for the procedure
        - model (torch.nn.Module): PyTorch model to train
        - optimizer (torch.optim.Optimizer): optimizer to use during training

        Returns:
        - final_ppl (float): perplexity of the best model on training set
        """  

        # Initialize a variable to store the best perplexity (PPL) value
        logs = []
        saved_weights = []
        T = 0
        P = 3
        t = 0
        best_ppl = math.inf
        PPL = math.inf
        pbar = tqdm(range(0, parameters['n_epochs']))  # Inizio da 0
        for epoch_K in pbar:
            
            loss = train_loop(parameters['train_loader'], optimizer, parameters['criterion_train'], model, parameters['clip'])    
                        
            if epoch_K % parameters['logging_interval'] == 0:  # Controlla se questa condizione è mai vera                      
                # Evaluate the model on the validation set
                PPL, _ = eval_loop(parameters['dev_loader'], parameters['criterion_eval'], model)
                logs.append(PPL)
                t = t + 1

                # If the current PPL is better than the best PPL
                if t > parameters['non_monotonic_interval'] and PPL > min(logs[:t-parameters['non_monotonic_interval']+1]): 
                    T = epoch_K

                if PPL < best_ppl:
                    best_ppl = PPL
                    best_model = copy.deepcopy(model).to('cpu')
                else:
                    P -= 1

                if P <= 0:
                    break

            if T != 0:
                saved_weights.append(copy.deepcopy(model.state_dict()))

            pbar.set_description(f'Epoch_K: {epoch_K} - PPL:{PPL} - T:{T} - t:{t}')

        # Average the saved weights to implement NT-AvSGD
        if T != 0 and saved_weights:
            avg_weights = average_weights(saved_weights)
            model.load_state_dict(avg_weights)
        elif best_model is not None:
            model = best_model

        model.to(DEVICE)
        torch.save(model.state_dict(), parameters['weight_path'])
        final_ppl,  _ = eval_loop(parameters['test_loader'], parameters['criterion_eval'], model)            
        return final_ppl

        
def load_model_and_print_ppl(model, parameters):
    """
    Load a model and print its Perplexity (PPL) on the test dataset.

    Parameters:
    - model (torch.nn.Module): untrained model architecture
    - model_path (str): path to the file containing the saved model parameters
    - test_loader (torch.utils.data.DataLoader): dataLoader for the test set
    - criterion_eval (torch.nn.Module): loss function used during evaluation
    - DEVICE (str or torch.DEVICE): DEVICE
    
    Returns:
    - ppl (float): perplexity of the model
    """
    
    # Check if the model file exists at the specified path.
    if not os.path.exists(parameters['weight_path']):
        print(f"\n-> No model file found at {parameters['weight_path']}.\n")
        return None
    print(f'\n-> Weights founded! Loading ...\n')

    model.load_state_dict(torch.load(parameters['weight_path'], map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    ppl, _ = eval_loop(parameters['test_loader'], parameters['criterion_eval'], model)
    
    return ppl

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        if model.variational_dropout:
                model.reset_vd_mask()

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

    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            if model.variational_dropout:
                model.reset_vd_mask()
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