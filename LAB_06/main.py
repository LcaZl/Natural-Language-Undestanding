# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions

    # Loading components
    print('\nRetrieving components ...\n')
    stanza.download("en")
    nltk.download('dependency_treebank')
    print('\nComponents loaded.\n')

    # Parameters and dataset
    samples = 10
    sentences = dependency_treebank.sents()[-samples:]
    test_graphs = dependency_treebank.parsed_sents()[-samples:]
    graph_info(test_graphs[-1], 'Treebanks')

    # Loading parsers

    # Parser conll configuration
    
    config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"DEPREL": {"nsubj": "subj"}}}
    
    spacy_parser = load_spacy_parser(config = config, verbose = False)
    stanza_parser = load_stanza_parser(config = config, verbose = False)
   
    # Parsing sentences with Spacy model
    spacy_result = parse_sentences(spacy_parser, 
                                   sentences, 
                                   parser_name='Spacy',
                                   verbose=True)
    spacy_graphs = [g for df, g in spacy_result]

    # Parsing sentences with Spacy_stanza model
    stanza_result = parse_sentences(stanza_parser, 
                                    sentences, 
                                    parser_name='Stanza',
                                    verbose=True)
    stanza_graphs = [g for df, g in stanza_result]

    # Comparing tags of the two parsers
    same_tags, diff_tags, diff_df = compare_tags(spacy_result, stanza_result)
    print(f"\n -> Number of same tags: {same_tags}")
    print(f" -> Number of different tags: {diff_tags}")
    print('-> First 10 differences:\n',tabulate(diff_df[:10], headers='keys', tablefmt='grid'), '\n')

    # Evaluating parsers against the ground truth
    spacy_score = evaluate_parser(spacy_graphs, test_graphs, parser_name='Spacy')
    stanza_score = evaluate_parser(stanza_graphs, test_graphs, parser_name = 'Stanza')

    print('LAS e UAS of each parser:\n')
    print(tabulate(pd.DataFrame([spacy_score, stanza_score]), headers='keys', tablefmt='grid'))