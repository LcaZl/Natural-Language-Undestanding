from functions import *

if __name__ == "__main__":
    
    # Parameters
    vocabulary_cutoff = 2
    ngram_order = 3
    train_size = 0.9
    name = 'shakespeare-macbeth.txt'#'shakespeare-macbeth.txt'
    predefined_tests = [
        {'phrase':'<UNK>', 'context':None},
        {'phrase':'<UNK>', 'context':['<UNK>']},
        {'phrase':'<UNK>', 'context':['the']},
        {'phrase':'go', 'context':['i']},
        {'phrase':'bid', 'context':None},
        {'phrase':'the king is dead', 'context':None},
        {'phrase':'welcome to you', 'context':None},
        {'phrase':'how are you', 'context':None}
    ]

    # Download and prepare dataset (List of lists of words)
    dataset_oov_sents = prepare_dataset(name)
    dataset_info(dataset_oov_sents, name, ngram_order, vocabulary_cutoff)

    # Splitting prepared dataset into train and test
    training_set, test_set = split_dataset(dataset_oov_sents, 0.8)

    # Replacing out of vocabulary words for both datasets. 
    # The vocabulary used is built only on train set.
    training_vocab = get_vocabulary(training_set, vocabulary_cutoff)
    training_set = replace_oov(training_set, training_vocab)
    test_set = replace_oov(test_set, training_vocab)

    # Train and test set information
    dataset_info(training_set, 'Training Set', ngram_order, vocabulary_cutoff)
    dataset_info(test_set, 'Test Set', ngram_order, vocabulary_cutoff)

    # 1 - Train and test Maximum Likelihood Estimator
    lm_mle = train_model('MLE', 
                         training_set, 
                         ngram_order)
    
    mle_tested_sents, mle_scores = test_model(lm_mle, 
                                              'MLE',
                                             test_set, 
                                             predefined_tests, 
                                             ngram_order)
    print(mle_scores)

    # 2 - Train and test Lidstone
    gamma = 0.5 # = 1 (Laplace)
    lm_lid = train_model('Lidstone', 
                         training_set, 
                         ngram_order,
                         gamma = gamma)
    
    lid_tested_sents, lid_scores = test_model(lm_lid, 
                                              'Lidstone',
                                             test_set, 
                                             predefined_tests, 
                                             ngram_order)
    
    
    # 3 - Train and test Absolute Discounting Interpolated (ADI)
    discount = 0.75
    lm_adi = train_model('ADI', 
                         training_set, 
                         ngram_order,
                         discount = discount)
    
    adi_tested_sents, adi_scores = test_model(lm_adi, 
                                              'ADI',
                                             test_set, 
                                             predefined_tests, 
                                             ngram_order)
    
    # 4 - Train and test NLTK StupidBackoff
    alpha = 0.4
    lm_sb = train_model('StupidBackoff', 
                        training_set, 
                        ngram_order,
                        alpha = alpha)
    
    sb_tested_sents, sb_scores = test_model(lm_sb, 
                                            'StupidBackoff',
                                          test_set, 
                                          predefined_tests, 
                                          ngram_order)

    # 5 - Train and test my StupidBackoff implementation
    alpha = 0.4
    epsilon = 0.01
    lm_mysb = train_model('MyStupidBackoff',
                            training_set, 
                            ngram_order,
                            alpha = alpha,
                            epsilon = epsilon)
    
    mysb_tested_sents, mysb_scores = test_model(lm_mysb, 
                                                'MyStupidBackoff',
                                                test_set, 
                                                predefined_tests, 
                                                ngram_order)


    # Comparison
    print(f'\n\nperformance comparison:\n')
    print(pd.concat([mle_scores, lid_scores, adi_scores, sb_scores, mysb_scores]))
    print('\n\n')
