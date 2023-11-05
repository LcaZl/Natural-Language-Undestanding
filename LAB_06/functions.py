
import nltk
import stanza
import spacy_stanza
import spacy
import spacy_conll
import en_core_web_sm
import pandas as pd
from spacy.tokenizer import Tokenizer
from nltk.parse.dependencygraph import DependencyGraph
from nltk.corpus import dependency_treebank
from nltk.parse.dependencygraph import DependencyGraph
from tqdm import tqdm
from nltk.parse import DependencyEvaluator
from tabulate import tabulate
spacy.load('en_core_web_sm')

def load_spacy_parser(config = None, verbose = False):
    """
    Load Spacy parser.

    Args:
    - config (dict, optional): configuration for the CONLL formatter
    - verbose (bool, default=False): if True, displays the model's configuration

    Returns:
    - spacy_nlp: Spacy's nlp object
    """
    print('Loading Spacy parser ...')
    spacy_nlp = spacy.load("en_core_web_sm")
    if config is not None:
        spacy_nlp.add_pipe("conll_formatter", config=config, last=True)

    spacy_nlp.tokenizer = Tokenizer(spacy_nlp.vocab)  
    if verbose:
        show_model_config(spacy_nlp, 'Spacy')

    print('Spacy parser loaded!\n')
    return spacy_nlp

def load_stanza_parser(config = None, verbose = False):
    """
    Load Stanza parser.

    Args:
    - config (dict, optional): configuration for the CONLL formatter
    - verbose (bool, default=False): if True, displays the model's configuration

    Returns:
    - stanza_nlp: Spacy_stanza's nlp object
    """
    print('Loading Stanza parser ...')
    stanza_nlp = spacy_stanza.load_pipeline("en", tokenize_pretokenized=True,
                                            download_method = None,
                                            verbose=False)
    if config is not None:
        config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"DEPREL": {"nsubj": "subj", "root":"ROOT"}}}
        stanza_nlp.add_pipe("conll_formatter", config=config, last=True)

    if verbose:
        show_model_config(stanza_nlp, 'Stanza')

    print('Stanza parser loaded!\n')
    return stanza_nlp

def parse_sentences(nlp, sentences, parser_name, verbose = False):
    """
    Parse sentences using the input parser.

    Args:
    - nlp: parser
    - sentences: list of sentences to parse.
    - parser_name: name of the parser being used.
    - verbose (bool, default=False): if True, displays the last sentence parsing process.

    Returns:
    - output: List of pair (Dataframe, DependencyGraph), one for each sentence.
    """
    output = []
    print(f'Parsing sentences with {parser_name} parser ...\n')

    for sent in tqdm(sentences):

        doc = nlp(' '.join(sent))
        df = doc._.pandas

        tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
        dg = DependencyGraph(tmp)

        output.append((df, dg))

    if verbose:
        print('\nLast sentence parsing info:\n')
        print(' - Dataframe with result of nlp():\n',df)
        graph_info(dg, parser_name)

    print('\n')
    return output

def compare_tags(p1, p2):
    """
    Compare tags between two parsers.

    Args:
    - p1: Parsed data with one parser
    - p2: Parsed data with another parser

    Returns:
    - diff_df: DataFrame containing differences in tags between parsers.
    """
    same_tags = 0
    different_tags = 0
    different_tags_list = []
    res_cols = ['LEMMA','DEPREL','UPOS','XPOS']
    print('Comparing parsers tags ...\n')
    
    for (df1, _), (df2, _) in zip(p1, p2):
        for ((l1, d1, u1, x1), (l2, d2, u2, x2)) in zip(df1[res_cols].values, df2[res_cols].values):
            if d1 == d2:
                same_tags += 1
            else:
                different_tags += 1
                different_tags_list.append({"Word": f'{l1}-{l2}', 
                                            "DEPREL_Spacy": d1, 
                                            "DEPREL_Stanza": d2,
                                            "UPOS Spacy": u1,
                                            "UPOS Stanza": u2,
                                            "XPOS Spacy": x1,
                                            "XPOS Stanza": x2})
    
    diff_df = pd.DataFrame(different_tags_list)

    return same_tags, different_tags, diff_df


def evaluate_parser(graphs, test, parser_name):
    """
    Evaluate a parser using DependencyEvaluator.

    Args:
    - graphs: list of parsed sentence graphs
    - test: ground truth parses of the same sentences
    - parser_name: name of the parser being evaluated

    Returns:
    - Dictionary with parser name and its LAS and UAS scores.
    """
    evaluator = DependencyEvaluator(graphs, test)
    las, uas =  evaluator.eval()

    return {'Parser':parser_name, 'LAS':las, 'UAS':uas}

def graph_info(graph, id):
    """
    Display information about a dependency graph.

    Args:
    - graph: Dependency graph object
    - id: label of correlated parsing type
    """
    print(f'[{id}] Sentence graph info:\n')
    print(f' - Graph nodes ')
    for node in graph.nodes.items():
        print(' - ', node)

    print('\n - Sentence parse Tree:\n')
    graph.tree().pretty_print(unicodelines = True, nodedist = 3)



def show_model_config(nlp, id):
    """
    Display configuration details of an NLP model.

    Args:
    - nlp: NLP object (Spacy or Stanza)
    - id: label of correlated parsing type
    """
    print(f'\n{id} model info:\n')
    for key, items in nlp.config.items():
        print(f' - {key}: {items}')