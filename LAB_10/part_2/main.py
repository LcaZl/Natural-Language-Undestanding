# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    model = BertModel.from_pretrained("bert-base-uncased")
    training_set, validation_set, test_set, lang = load_dataset()
    
    # Dataloader instantiation
    train_loader = DataLoader(training_set, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(validation_set, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=64, collate_fn=collate_fn)


    experiments = {
        'Experiment_1': {
            'model':'BERT',
            'train_loader':train_loader,
            'dev_loader':dev_loader,
            'test_loader':test_loader,
            'num_slot_labels':len(lang.slot2id),
            'num_intent_labels':len(lang.intent2id),
            'vocabulary_size':len(lang.tokenizer.vocab),
            'lang':lang,
            'epochs':20,
            'patience':3,
            'clip':5,
            'runs':10,
            'learning_rate':5e-5,
            'dropout_probabilities':0.1,
            'slot_loss_coefficient': 0.9,
            'use_crf':True,
            'criterion_slots' : nn.CrossEntropyLoss(ignore_index = PAD_TOKEN),
            'criterion_intents' : nn.CrossEntropyLoss() },

        'Experiment_2': {
            'model':'BERT',
            'train_loader':train_loader,
            'dev_loader':dev_loader,
            'test_loader':test_loader,
            'num_slot_labels':len(lang.slot2id),
            'num_intent_labels':len(lang.intent2id),
            'vocabulary_size':len(lang.tokenizer.vocab),
            'lang':lang,
            'epochs':20,
            'patience':3,
            'clip':5,
            'runs':10,
            'learning_rate':5e-5,
            'dropout_probabilities':0.1,
            'slot_loss_coefficient': 0.9,
            'use_crf':False,
            'criterion_slots' : nn.CrossEntropyLoss(ignore_index = PAD_TOKEN),
            'criterion_intents' : nn.CrossEntropyLoss() }
            
    }

    results = execute_experiments(experiments)
    print('\nExperiments comparison:\n')
    print(tabulate(results, headers='keys', tablefmt='grid', showindex=True))
