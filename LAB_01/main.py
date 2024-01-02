from functions import *

if __name__ == "__main__":

    # Book, frequency list and cut-off parameters
    book = 'milton-paradise.txt'
    N = 12 # Top N items of frequency list to show
    cut_off_bounds = [2,100] # Lower bound and upper bound for frequency cut-off
    stop_words_sets = ['None', 'NLTK', 'Spacy', 'Scikit']
    libraries = ['Ref', 'NLTK', 'Spacy']
    frequency_list_methods = ['Counter', 'NLTK']
    combinations = len(stop_words_sets) * len(libraries) * len(frequency_list_methods)

    # Initialization of final result's objects

    lx_list = [] # Lexicons for each configuration
    all_stats = pd.DataFrame() # Statistics for each configuration
    all_frequency = pd.DataFrame() # Top-N frequency list for each configuration

    # Set to true to see intermediate results while computing
    show_progress = False

    # Computing statistics
    iteration = 1
    for lib in libraries: # Library: 1-References, 2-NLTK, 3-Spacy.
        
        for stop_words_set in stop_words_sets: # Stop Words methods: 1-None, 2-NLTK, 3-Spacy, 4-Scikit-learn.
            
            for fr_list_method in frequency_list_methods: # Frequency list methods: 1-Collection.Counter, 2-NLTK.
                
                print(f'\n--> Analyzing configuration {iteration}/{combinations}')
                print(f'----> Library: {lib}')
                print(f'----> Stop words set: {stop_words_set}')
                print(f'----> Frequency list method: {fr_list_method}')
                
                lx, stats, freq_list = compute_descriptive_stats(book = book,
                                                                library = lib, 
                                                                stop_word_set = stop_words_set, 
                                                                frequency_list_method = fr_list_method, 
                                                                nbest_v = N, 
                                                                cut_off_bounds = cut_off_bounds,
                                                                show_progress = show_progress)
                
                # Concatenate actual output with the existing ones
                lx_list.append({stats.index.values[0] : lx})
                all_stats = pd.concat([all_stats, stats], ignore_index = False)
                all_frequency = pd.concat([all_frequency, freq_list])
                iteration += 1
                
    # Show final results
    print(f'\n------ Corpus, Lexicon and frequency list information for each configuration ------\n')
    print('-> Index structure: <LIBrary, Stop Words Set, Frequency List Method>')
    print_results(all_stats, 'Corpus and Lexicon statistics')
    print_results(all_frequency, f'Frequency list top-{N}')
