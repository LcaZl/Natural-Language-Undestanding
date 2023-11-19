# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
    training_set, validation_set, test_set, lang = load_dataset()
    
    # Dataloader instantiation
    train_loader = DataLoader(training_set, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(validation_set, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=64, collate_fn=collate_fn)
    
    experiment_base = {
            'model':'IAS',
            'train_loader':train_loader,
            'dev_loader':dev_loader,
            'test_loader':test_loader,
            'clip':5,
            'output_slots':len(lang.slot2id),
            'output_intent':len(lang.intent2id),
            'vocabulary_size':len(lang.word2id),
            'epochs':200,
            'patience':3,
            'lang':lang,
            'runs':5,
    }


    experiments = {
        'Experiment_1':{
            'hidden_layer_size':300,
            'embedded_layer_size':300,
            'learning_rate':0.001,
            'dropout_probabilities':0,
            'bidirectional':False,
            **experiment_base
        },
        
        'Experiment_2':{
            'hidden_layer_size':300,
            'embedded_layer_size':300,
            'learning_rate':0.001,
            'dropout_probabilities':0,
            'bidirectional': True,
            **experiment_base
        },

        'Experiment_3':{
            'hidden_layer_size':300,
            'embedded_layer_size':300,
            'learning_rate':0.001,
            'dropout_probabilities':0.05,
            'bidirectional': True,
            **experiment_base
        }
    }

    best_model1, reports, losses1 = execute_experiment('Experiment_1', experiments['Experiment_1'])
    scores = get_scores(reports, 'Experiment_1')

    best_model2, reports, losses2 = execute_experiment('Experiment_2', experiments['Experiment_2'])
    scores = pd.concat([scores, get_scores(reports, 'Experiment_2')], axis=0)

    best_model3, reports, losses3 = execute_experiment('Experiment_3', experiments['Experiment_3'])
    scores = pd.concat([scores, get_scores(reports, 'Experiment_3')], axis=0)

    print('Experiments:\n')
    print(tabulate(scores, headers='keys', tablefmt='grid', showindex=True))


"""
    cols = ['Id','Run','Accuracy','Accuracy Std','F score', 'F Std']
    scores = pd.DataFrame(columns = cols)
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
"""