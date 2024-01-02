from functions import *

if __name__ == "__main__":

    #Wrtite the code to load the datasets and to run your functions

    training_set, test_set = dataset()
    dataset_info(training_set, test_set)

    print(f'0 - Training CRF with "Baseline" features schema\n')
    c_training_set, c_test_set = feature_selection(training_set.copy(), test_set.copy(), 'Baseline')
    crf_0 = train_crf(c_training_set)
    score_0 = test_crf(crf_0, c_test_set)

    print(f'1 - Training CRF with Spacy features schema\n')
    c_training_set, c_test_set = feature_selection(training_set.copy(), test_set.copy(), 'Spacy_1')
    crf_1 = train_crf(c_training_set)
    score_1 = test_crf(crf_1, c_test_set)

    print(f'2 - Training CRF with Spacy plus "suffix_ "features schema\n')
    c_training_set, c_test_set = feature_selection(training_set.copy(), test_set.copy(), 'Spacy_2')
    crf_2 = train_crf(c_training_set)
    score_2 = test_crf(crf_2, c_test_set)

    print(f'3 - Training CRF with Conll Tutorial features schema - Features window [0,0]\n')
    c_training_set, c_test_set = feature_selection(training_set.copy(), test_set.copy(), 'ConllTutorial')
    crf_3 = train_crf(c_training_set)
    score_3 = test_crf(crf_3, c_test_set)

    print(f'4 - Training CRF with Conll Tutorial features schema - Features windows[-1,+1]\n')
    c_training_set, c_test_set = feature_selection(training_set.copy(), test_set.copy(), 'FeatureWindow_1')
    crf_4 = train_crf(c_training_set)
    score_4 = test_crf(crf_4, c_test_set)

    print(f'5 - Training CRF with Conll Tutorial features schema - Features window [-2,+2]\n')
    c_training_set, c_test_set = feature_selection(training_set.copy(), test_set.copy(), 'FeatureWindow_2')
    crf_5 = train_crf(c_training_set)
    score_5 = test_crf(crf_5, c_test_set)

    # Comparison
    all_scores = [score_0, score_1, score_2, score_3, score_4, score_5]
    labels = ["Baseline", "Spacy_1", "Spacy_2", "ConllTutorial", "FeatureWindow_1", "FeatureWindow_2"]    
    final_scores = []

    # Creating comparison dataframe
    for idx, (score, label) in enumerate(zip(all_scores, labels)):
        score['Config'] = [f'{idx}-{label}-{entity}' for entity in score.index]
        score['Entity'] = [f'{entity}' for entity in score.index]
        score['Idx'] = [f'{idx}' for entity in score.index]
        final_scores.append(score)

    final_df = pd.concat(final_scores, ignore_index=True).set_index('Config')
    print(tabulate(final_df.sort_values(by=['Entity','Idx'])[['Precision', 'Recall', 'F1-Score', 'Support']], 
                   headers='keys', 
                   tablefmt='grid', 
                   showindex=True))