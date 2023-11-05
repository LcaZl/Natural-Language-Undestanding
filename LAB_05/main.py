# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    # Wrtite the code to load the datasets and to run your functions
    # Print the results

    # Probabilistic Context-Free Grammar
    grammar_rules = [
        'S -> NP VP [1.0]',
        
        'NP -> Det N [0.3]',
        'NP -> Det N PP [0.4]',
        'NP -> Det N N PP [0.15]',
        'NP -> Det N N [0.15]',

        'VP -> V NP [0.5]',
        'VP -> V NP PP [0.5]',
        
        'PP -> P NP [0.5]',
        'PP -> P N [0.5]',
        
        'Det -> "The" [0.5]',
        'Det -> "the" [0.5]',
        
        'N -> "chef" [0.07]',
        'N -> "kitchen" [0.07]',
        'N -> "meal" [0.07]',
        'N -> "care" [0.07]',
        'N -> "waiter" [0.07]',
        'N -> "customer" [0.09]',
        'N -> "dining" [0.07]',
        'N -> "table" [0.07]',
        'N -> "entrance" [0.07]',
        'N -> "food" [0.07]',
        'N -> "dessert" [0.07]',
        'N -> "rapidly" [0.07]',
        'N -> "atmosphere" [0.07]',
        'N -> "room" [0.07]',

        'V -> "cooked" [0.25]',
        'V -> "enjoyed" [0.25]',
        'V -> "welcome" [0.25]',
        'V -> "served" [0.25]',

        'P -> "in" [0.25]',
        'P -> "with" [0.25]',
        'P -> "at" [0.25]',
        'P -> "by" [0.25]',

    ]

    # Test sentences
    sentences = [
        "The chef in the kitchen cooked the meal with care",
        "The waiter in the dining room served the food rapidly",
        "The customer at the table enjoyed the dessert",
        "The customer by the entrance enjoyed the atmosphere"
    ]

    #tree_test()

    # Evaluate grammar with different sentences and parsers
    parsers = ['Viterbi', 'Chart', 'InsideChart', 'RandomChart']
    PCFG = nltk.PCFG.fromstring(grammar_rules)
    grammar_info(PCFG)
    evaluate_grammar(PCFG, sentences, parsers)

    # Generating sents with NLTK
    print(f'Generating sentences with nltk.generate.generate()')
    start = Nonterminal('S')
    n = 10
    depth = 5

    for sent in nltk_generate(PCFG, n=n, depth=depth):
        print('-',sent)

    # Generating sents with PCFG
    print(f'\n - Generate sentences with PCFG module:')
    PCFG_OM = pkg_pcfg.fromstring(grammar_rules)
    n = 10

    for sent in pkg_pcfg.generate(PCFG_OM, n=n):
        print('-',sent)
    print('\n')