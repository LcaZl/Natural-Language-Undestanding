# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
"""
Lab Exercise

Same test set for all the experiments, you can use K-fold validation

    Extend collocational features with
        POS-tags
        Ngrams within window
    Concatenate BOW and new collocational feature vectors & evaluate
    Evaluate Lesk Original and Graph-based (Lesk Similarity or Pedersen) metrics on the same test split and compare

"""
from functions import *

if __name__ == "__main__":

    # Dataset used for WSD, each element is composed by:
    # - word: the target word to disambiguate
    # - Position: postion of the word in the phrase of the context
    # - context: phrase in which the word appear. (each word is a tuple (word, Pos Tag)) (pos tag -> type of word (noun, verb, ecc))
    # - sense: sense of the word in the context

    wsd_dataset = senseval.instances('interest.pos')
    dataset_info(wsd_dataset)
    
    # Generating features
    print(f'\nGenerating features ...')

    windows = [1,2,3]
    ngrams = [1,2,3]

    labels = get_labels(wsd_dataset)
    bow_vector = bow_features(wsd_dataset)

    collocational_feat_vect = lab_collocational_features(wsd_dataset)
    postag_vectors = { f'postag_[-{w},+{w}]' : pos_tag_features(wsd_dataset, w) for w in windows}
    ngram_vectors = { f'ngram_[-{w},+{w}]_n{n}' : ngram_features(wsd_dataset, w, n) for w in windows for n in ngrams}

    print('Features generated.')
    
    # Features check
    feats_check = check_features([{'Context' : bow_vector}, {'Lab':collocational_feat_vect}, postag_vectors, ngram_vectors])
    print(f'\n-> Feature sanity check status: {feats_check}\n')
    if not feats_check:
        print('Feature check not passed. Exiting.\n')
        exit()

    # Experiments

    print('\nTraining ...')
    experiments = []

    # Training as in lab_08 - Only collocational features
    print(' -> Training with fixed collocational features ...')
    experiments.append(train_model(collocational_feat_vect, labels, features_id = f'Fixed collocational'))

    # Training as in lab_08 - collocational and BOW features
    print(' -> Training with BOW and fixed collocational features ...')
    curr_features = concatenate_feature_vectors([bow_vector, collocational_feat_vect])
    experiments.append(train_model(curr_features, labels, features_id = f'BOW and collocational Features'))

    # Training using only BOW features
    print(' -> Training with BOW features ...')
    experiments.append(train_model(bow_vector, labels, features_id = f'BOW'))

    # Training with different windows for pos tags
    print(' -> Training with postag features ...')
    for postag_id, postag_vector in postag_vectors.items():
        experiments.append(train_model(postag_vector, labels, features_id = postag_id))

    # Training with ngrams in different windows
    print(' -> Training with grams features ...')
    for ngram_id, ngram_vector in ngram_vectors.items():
        experiments.append(train_model(ngram_vector, labels, features_id = ngram_id))

    # Training with BOW and new features
    print(' -> Training with BOW, posttag and ngram features ...')
    for postag_id, postag_vector in postag_vectors.items():

        for ngram_id, ngram_vector in ngram_vectors.items():
            curr_features = concatenate_feature_vectors([bow_vector, postag_vector, ngram_vector])
            experiments.append(train_model(curr_features, labels, features_id = f'BOW + {postag_id} + {ngram_id}'))

    # Training simplified Lesk
    print(' -> Training with simplified Lesk ...')
    lsmp_score, lsmp_df = wsd_evaluation(wsd_dataset, 'LeskSimp')
    experiments.append(lsmp_score)

    # Training orginal Lesk
    print(' -> Training with Lesk original ...')
    lo_score, lo_df = wsd_evaluation(wsd_dataset, 'LeskOr')
    experiments.append(lo_score)

    # Training Lesk with similarity
    print(' -> Training with Lesk similarity ...')
    lsml_score, lsml_df = wsd_evaluation(wsd_dataset, 'LeskSimi')
    experiments.append(lsml_score)

    print('Models trained.\n')

    # Experiments comparison
    print(f'\nExperiment results:\n')

    experiments_score = pd.DataFrame(data=experiments, columns=['Model','Features used','Accuracy','Precision','Recall','F1-Measure'])
    best_score = experiments_score.loc[experiments_score['Accuracy'].idxmax()]

    print(f' - Best score obtained with', best_score['Model'], '(Features:', best_score['Features used'], ') -> ', best_score['Accuracy'])
    print(f' - All experiments score:\n')
    print(tabulate(experiments_score, headers='keys', tablefmt='grid', showindex=True))

    print(f'\n - More details about performance of all versions fo Lesk and Pederson:\n')    
    more_info = pd.concat([lsmp_df, lo_df, lsml_df], ignore_index = False, axis = 1)
    print(tabulate(more_info, headers='keys', tablefmt='grid', showindex=True))