from functions import *

if __name__ == "__main__":

    # Test sentences
    sentences = [
        "The chef in the kitchen cooked the meal with care",
        "The waiter in the dining room served the food rapidly",
        "The customer at the table enjoyed the dessert",
        "The customer by the entrance enjoyed the atmosphere"
    ]

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

    tree_test()

    # Evaluate grammar with different sentences and parsers
    parsers = ['Viterbi', 'Chart', 'InsideChart', 'RandomChart']
    myPCFG = nltk.PCFG.fromstring(grammar_rules)
    grammar_info(myPCFG)
    evaluate_grammar(myPCFG, sentences, parsers)

    # Generate phrases with nltk
    nltk_generate_phrases(myPCFG,Nonterminal('VP'), 20, 5)
    nltk_generate_phrases(myPCFG,Nonterminal('NP'), 20, 10)
    nltk_generate_phrases(myPCFG,Nonterminal('PP'), 20, 7)
    nltk_generate_phrases(myPCFG,Nonterminal('N'), 20, 3)

    # Generating sents with PCFG module
    print(f'\n - Generate sentences with PCFG module:')
    PCFG_OM = pkg_pcfg.fromstring(grammar_rules)
    n = 10

    for sent in pkg_pcfg.generate(PCFG_OM, n=n):
        print('-',sent)
    print('\n')