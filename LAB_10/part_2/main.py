from functions import *

if __name__ == "__main__":

    model = BertModel.from_pretrained("bert-base-uncased")
    training_set, validation_set, test_set, lang = load_dataset()
    
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
            'epochs':200,
            'patience':3,
            'clip':5,
            'runs':5,
            'learning_rate':5e-5,
            'dropout_probabilities':0.1,
            'use_crf':False,
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
            'epochs':200,
            'patience':3,
            'clip':5,
            'runs':5,
            'learning_rate':5e-5,
            'dropout_probabilities':0.1,
            'use_crf':True,
            'criterion_slots' : nn.CrossEntropyLoss(ignore_index = PAD_TOKEN),
            'criterion_intents' : nn.CrossEntropyLoss() }
            
    }

    cols = ['Id','Run','Accuracy','Accuracy Std.','F-score', 'F-score Std.']
    scores = pd.DataFrame(columns = cols)

    reports, best_model1, losses1 = experiment('Experiment_1', experiments['Experiment_1'])
    scores = get_scores(reports, 'Experiment_1')
    plot_aligned_losses(losses1[0][f'run_{best_model1[1][0]}'], 
                        losses1[1][f'run_{best_model1[1][0]}'], 
                        'Experiment 1 - Best model losses')
    
    reports, best_model2, losses2 = experiment('Experiment_2', experiments['Experiment_2'])
    scores = pd.concat([scores, get_scores(reports, 'Experiment_2')], axis=0)
    plot_aligned_losses(losses2[0][f'run_{best_model2[1][0]}'], 
                        losses2[1][f'run_{best_model2[1][0]}'], 
                        'Experiment 2 - Best model losses')
    
    print('Experiments:\n')
    print(tabulate(scores, headers='keys', tablefmt='grid', showindex=True))
