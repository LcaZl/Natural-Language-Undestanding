import nltk
from nltk.tree import Tree
from nltk import Nonterminal
from nltk.parse.generate import generate as nltk_generate
from pcfg import PCFG as pkg_pcfg
from pprint import pprint

def generate_parsing_trees(grammar, sentence, parser_type):
    """
    Generates parsing trees using input grammar and sentence.
    The parser will be instantieted accordingly to the input type.

    Parameters:
    - grammar (nltk.grammar): grammar based on which parsing will be done
    - sentence (str): input sentence to be parsed
    - parser_type (str): type of parser to be used

    Returns:
    - list: list of parsing tree
    """

    parser = None

    if parser_type == 'Viterbi':
        print(f'\n----> Viterbi Parser:')
        parser = nltk.ViterbiParser(grammar) 

    elif parser_type == 'Chart':
        print(f'\n----> Chart Parser')
        parser = nltk.ChartParser(grammar)

    elif parser_type =='InsideChart':
        print(f'\n----> Inside Chart parser:')
        parser = nltk.InsideChartParser(grammar, beam_size=100)

    elif parser_type == 'RandomChart':
        print(f'\n----> Random Chart Parser:')
        parser = nltk.RandomChartParser(grammar, beam_size=100)

    trees = list(parser.parse(sentence.split()))
    
    return trees
     
def evaluate_grammar(grammar, sentences, parsers):
    """
    Validate grammar by parsing multiple sentences with different parsers and print
    the results of the process.

    Parameters:
    - grammar (nltk.grammar): grammar to be evaluated
    - sentences (list of str): list of sentences to be parsed
    - parsers (list of str): list of parser types to use for parsing
    """
    for sent in sentences:

        for parser_type in parsers:

            trees = generate_parsing_trees(grammar, sent, parser_type)
            print(f'----> {len(trees)} trees generated.')
            print(f'----> Sentence: {sent}')
            print_trees(trees)

def grammar_info(grammar):
    """
    Prints information about the input grammar.

    Parameters:
    - grammar (nltk.grammar): grammar
    """
    print(f'Grammar info:')
    print(f' - Start symbol: {grammar.start()}')
    print(f' - Productions:\n', grammar, '\n\n')
    # filter by the left-hand side or the first item in the right-hand side (lhs, rhs)
    # Nonterminal('NP')
    #print(f'CFG Productions:', grammar.productions(lhs=None, rhs=None, empty=False))

def print_trees(trees):
    """
    Prints information and visualization of a list of trees.
    Prints information like:
     - leaves, 
     - flattened representation, 
     - POS tags, 
     - height,
     - visual representation of each tree.

    Parameters:
    - trees (list of nltk.Tree): list of trees to be printed

   
    """
    for i, tree in enumerate(trees):
        print(f'\n - Tree n.{i+1}:')
        print(f' - Leaves: {tree.leaves()}')
        print(f' - Flatten: {tree.flatten()}')
        print(f' - Pos tag: {tree.pos()}')
        print(f' - Height: {tree.height()}\n')
        tree.pretty_print(unicodelines=True, nodedist=5)


def nltk_generate_phrases(grammar, start, n , depth):
    """
        Generating sents with NLTK
    """
    print(f'\nGenerating sentences with nltk.generate.generate()')
    print(f'N: {n} - Start: {start} - Depth: {depth}')
    for sent in nltk_generate(grammar, start=start, n=n, depth=depth):
        print('-',sent)


def tree_test():
        
        parse_tree_str = "(S (NP (PRON I)) (VP (V saw) (NP (Det the) (N man)) (PP (P with) (NP (Det a) (N telescope)))))"
        tree = Tree.fromstring(parse_tree_str)
        print(f'\nTree info:\n')

        print(f'Tree .pprint():\n')
        tree.pprint()

        print(f'\nTree pretty_print(unicodelines=True, nodedist=4):\n')
        tree.pretty_print(unicodelines=True, nodedist=5) 

        # Productions
        print('Productions:',tree.productions())

        # Label
        print('Label:', tree[1].label())

        # Pos tags
        print('Pos Tags:', tree.pos())

        # Leaves 
        print('Leaves:', tree.leaves())

        # Flatten
        print('Flatten:', tree.flatten() )
