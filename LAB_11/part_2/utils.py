# Add functions or classes used for data loading and preprocessing
import string
# Parameters
PAD_TOKEN = 0
UNK_TOKEN = 1
DEVICE = 'cuda:0'
INFO_ENABLED = False
MAX_VOCAB_SIZE = 10000
TRAIN_PATH = 'dataset/laptop14_train.txt'
TEST_PATH = 'dataset/laptop14_test.txt'

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = file.read()
    dataset = dataset.split('\n')
    dataset = [el.strip() for el in dataset]
    return dataset

def process_raw_data(dataset):
    new_dataset = []
    for sample in dataset:
        if sample:
            raw_sent, words_tagged = sample.split('####')
            
            words_tagged = words_tagged.split()
            tags = [w.split('=')[1] for w in words_tagged]
            words = [w.split('=')[0] for w in words_tagged]
            sent = ' '.join(words)

            assert len(words) == len(tags)

            new_dataset.append({'sentence':sent, 'words':words, 'tags':tags})

    return new_dataset

def load_dataset():
    print(f'Loading Dataset Laptop 14...')

    train_raw = read_file(TRAIN_PATH)
    test_raw = read_file(TEST_PATH)

    train_set = process_raw_data(train_raw)
    test_set = process_raw_data(test_raw)

    print(' - Training sents:', len(train_raw))
    print(' - Test sents:', len(test_raw))
    print(' - Raw sent:', train_raw[0])
    print(f' - Training len:', len(train_set))
    print(f' - Train sample:', train_set[0])
    print(f' - Test len:', len(test_set))
    print(f' - Test sample:', test_set[1])


