from functions import *

if __name__ == "__main__":

    # Experimemts
    experiments  = {

        'Experiment_1' : {
        'Description' : 'Count Vectorization Test',
        'vectorization_method' : 'CountVect',
        'strip_accents' : None,
        'lowercase' : True,
        'stop_words' : None,
        'max_df' : 1.0, # 1.0 default
        'min_df' : 1, # 1 default
        'max_features' : None
        },

        'Experiment_2' : {
        'Description' : 'TF-IDF Transformation Test',
        'vectorization_method' : 'TF_IDF',
        'strip_accents' : None,
        'lowercase' : True,
        'stop_words' : None,
        'max_df' : 1.0, # 1.0 default
        'min_df' : 1, # 1 default
        'max_features' : None
        },

        'Experiment_3' : {
        'Description' : 'TF-IDF with min e max cut-offs set',
        'vectorization_method' : 'TF_IDF',
        'strip_accents' : None,
        'lowercase' : True,
        'stop_words' : None,
        'max_df' : 2000, # 1.0 default
        'min_df' : 5, # 1 default
        'max_features' : None
        },

        'Experiment_4' : {
        'Description' : 'TF-IDF with stop words removed.',
        'vectorization_method' : 'TF_IDF',
        'strip_accents' : None,
        'lowercase' : True,
        'stop_words' : 'english',
        'max_df' : 1.0, # 1.0 default
        'min_df' : 1, # 1 default
        'max_features' : None
        },

        'Experiment_5' : {
        'Description' : 'TF-IDF without lowercasing (default=True)',
        'vectorization_method' : 'TF_IDF',
        'strip_accents' : None, # None default 
        'lowercase' : False, # True default
        'stop_words' : None, # None default
        'max_df' : 1.0, # 1.0 default
        'min_df' : 1, # 1 default
        'max_features' : None # None default
        },
    }   

    # Dataset, validation methods, baseline strategies and evaluation metrics
    cross_validation_methods = ['kfold','skfold']
    cross_validation_folds = 5
    baseline_strategies = ['uniform','stratified','most_frequent']
    evaluation_scores = ['accuracy','f1_weighted','recall_weighted']

    # Loading dataset
    
    dataset = fetch_20newsgroups(
                            subset='all', 
                            remove=('headers', 'footers', 'quotes'),
                            categories = ['alt.atheism',
                                    'comp.graphics',
                                    #'comp.os.ms-windows.misc',
                                    #'comp.sys.ibm.pc.hardware',
                                    #'comp.sys.mac.hardware',
                                    #'comp.windows.x',
                                    'misc.forsale',
                                    'rec.autos',
                                    #'rec.motorcycles',
                                    #'rec.sport.baseball',
                                    #'rec.sport.hockey',
                                    'sci.crypt',
                                    #'sci.electronics',
                                    #'sci.med',
                                    #'sci.space',
                                    'soc.religion.christian',
                                    #'talk.politics.guns',
                                    'talk.politics.mideast'
                                    #'talk.politics.misc',
                                    #'talk.religion.misc'
                                ]
                        )
 
    # Dataset info
    dataset_info(dataset)
    print(f' - Cross-validation methods: {cross_validation_methods}')
    print(f' - Cross validation folds: {cross_validation_folds}')
    print(f' - Baseline strategies:', baseline_strategies)
    print(f' - Evaluation metrics: {evaluation_scores}\n')

    # Text pre-processing
    print(f'Normalizing dataset ...\n')
    dataset.data = [normalize_document(doc) for doc in tqdm(dataset.data)]
    print(f'\nDataset normalized!\n')

    # Initialize the dataframe that will contain the average scores of each experiment.
    reports_comparison = pd.DataFrame()

    # Executing experiments
    for (experiment_id, experiment_parameters) in experiments.items():
        
        # Current experiment info
        print(f'\n{experiment_id}')
        folds_report, exp_report = execute_experiment(id = experiment_id, 
                           parameters = experiment_parameters, 
                           dataset = dataset, 
                           cv_methods = cross_validation_methods,
                           cv_folds = cross_validation_folds,
                           baseline_st = baseline_strategies,
                           metrics = evaluation_scores
                           )
    
        reports_comparison = pd.concat([reports_comparison, exp_report], axis = 1)

    # Compare each experiment average scores
    print(f'\nExperiments results:\n')
    for (exp_id, exp_par) in experiments.items():
        print(f' - {exp_id}:', exp_par['Description'])
    print_results(reports_comparison, ' - Comparison:\n')
