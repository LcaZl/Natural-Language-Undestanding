# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import pandas as pd
import math
import copy
import os
import math
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import torch.optim as optim
from torch.utils.data import DataLoader

from functools import partial
from sklearn.metrics.pairwise import cosine_similarity

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

            model = LangModel(
                        experiment_parameters['model_name'],
                        experiment_parameters['embedded_layer_size'], 
                        experiment_parameters['hidden_layer_size'], 
                        experiment_parameters['vocab_len'], 
                        pad_index = experiment_parameters['lang'].word2id["<pad>"],
                        out_dp_prob = experiment_parameters['output_dropout'],
                        emb_dp_prob = experiment_parameters['embedding_dropout']).to(DEVICE)
            
            model.apply(init_weights)

            # Defining the optimizer
            optimizer = None
            if experiment_parameters['optmz_type'] == 'SGD':
                optimizer = optim.SGD(model.parameters(), 
                                    lr=experiment_parameters['optmz_learning_rate'])
            elif experiment_parameters['optmz_type'] == 'Adam':
                optimizer = optim.AdamW(model.parameters(), 
                                            lr=experiment_parameters['optmz_learning_rate'])
                
            # If weights of the model altready exists, don't train.
            model_score = load_model_and_print_ppl(model, experiment_parameters)
            
            if model_score is None: # If the weight in specified folder doesn't exists
                print(f'\nStart training ... \n')
                model_score = train_lm(experiment_parameters,  model, optimizer)

            
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
        - DEVICE (str or torch.DEVICE): DEVICE, "cpu" or "cuda"
        - clip (float): value to clip gradients during training

        Returns:
        - final_ppl (float): perplexity of the best model on training set
        """        
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1,parameters['n_epochs']))
        P = parameters['patience']
        for epoch in pbar:
            loss = train_loop(parameters, optimizer, model)    
            
            if epoch % 1 == 0:
                ppl_dev, _ = eval_loop(parameters['dev_loader'], parameters, model)
                pbar.set_description("PPL: %f" % ppl_dev)
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    #P = parameters['patience']
                else:
                    P -= 1
                    
                if P <= 0: 
                    break
                                
        best_model.to(DEVICE)
        torch.save(best_model.state_dict(), parameters['weight_path'])
        final_ppl,  _ = eval_loop(parameters['test_loader'], parameters, best_model)            
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
    ppl, _ = eval_loop(parameters['test_loader'], parameters, model)
    
    return ppl

## from LAB_09

def train_loop(parameters, optimizer, model):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in parameters['train_loader']:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = parameters['criterion_train'](output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['clip'])  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, parameters, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = parameters['criterion_eval'](output, sample['target'])
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