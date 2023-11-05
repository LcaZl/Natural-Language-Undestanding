from functions import *
if __name__ == "__main__":
    
    # Mapping spacy tags to nltk tags
    spacy_to_nltk_map = {
        'ADJ': 'ADJ',
        'ADP': 'ADP',
        'ADV': 'ADV',
        'AUX': 'VERB',
        'CONJ': 'CONJ',
        'CCONJ': 'CONJ',
        'DET': 'DET',
        'INTJ': 'X',
        'NOUN': 'NOUN',
        'NUM': 'NUM',
        'PART': 'PRT',
        'PRON': 'PRON',
        'PROPN': 'NOUN',
        'PUNCT': '.',
        'SCONJ': 'CONJ',
        'SYM': 'X',
        'VERB': 'VERB',
        'X': 'X'
    }

    # Dataset settings
    training_set_size = 0.8

    # NgramTagger settings
    cutoff_settings = [1,2,3,4,5]
    backoff_settings = [True, False]
    N_setting = 4
    print(f'\nDownloading dataset ...')
    training_set, test_set = prepare_dataset(training_set_size)

    # NGramTagger experiments
    print(f'\nEvaluating NGramTagger with different settings:')
    nltk_scores = evaluate_nltk_taggers(
                            training_set = training_set,
                            test_set = test_set,
                            Nmax = N_setting,
                            backoffs = backoff_settings,
                            cutoffs = cutoff_settings)

    # Spacy evaluation
    print(f'Evaluating Spacy POS-tags')
    spacy_tagger = get_spacy_tagger(test_set, spacy_to_nltk_map)
    spacy_score = evaluate_spacy_tagger(spacy_tagger, test_set)

    # Comparison
    print(f'\nResults:')
    print(f' - NLTK NgramTagger scores of {len(nltk_scores)} configuration of parameters:\n',nltk_scores)
    nltk_bs = nltk_scores.loc[nltk_scores['Accuracy'].idxmax()]
    print(f'\n - NLTK best accuracy:', nltk_bs['Accuracy'],', (with N =', nltk_bs['N'],', backoff =', nltk_bs['Backoff'],', Cut-off =', nltk_bs['Cut-off'],')')
    print(f' - Spacy accuracy: {spacy_score}\n\n')

