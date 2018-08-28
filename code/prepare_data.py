import tensorflow as tf
import os
import numpy as np
import pandas as pd
import nltk
import pickle as pl
from functools import reduce 

def build_vocabulary_from_data(data):
    train = data['train']
    dev = data['dev']
    test = data['test']
    
    train['tokenized_sent']=train.apply(lambda row: set(row['tokenized_sent1']).union(set(row['tokenized_sent2'])), axis=1)
    dev['tokenized_sent']=dev.apply(lambda row: set(row['tokenized_sent1']).union(set(row['tokenized_sent2'])), axis=1)
    test['tokenized_sent']=test.apply(lambda row: set(row['tokenized_sent1']).union(set(row['tokenized_sent2'])), axis=1)
   
    train_token_set = reduce(set.union,
train['tokenized_sent'].tolist())
    print("words from train: "+str(len(train_token_set)))
    dev_token_set = reduce(set.union, dev['tokenized_sent'].tolist())
    print("words from dev: "+str(len(dev_token_set)))
    test_token_set = reduce(set.union, test['tokenized_sent'].tolist())
    print("words from test: "+str(len(test_token_set)))
    
    vocabulary_from_data = train_token_set.union(dev_token_set)
    print("how many words from data: "+str(len(vocabulary_from_data)))
    return vocabulary_from_data

def preprocess(data_file):
    data = pd.read_csv(data_file, usecols=['gold_label', 'sentence1',
'sentence2'], sep="\t", header='infer')    
    data['tokenized_sent1'] = data.apply(lambda row:
nltk.word_tokenize(str(row['sentence1']).lower()), axis=1)
    data['tokenized_sent2'] = data.apply(lambda row:
nltk.word_tokenize(str(row['sentence2']).lower()), axis=1)
    return data

def initialize(data_dir):
    snli_dir = os.path.join(data_dir, "snli_1.0")
    snli_train_file = os.path.join(snli_dir, "snli_1.0_train.txt")
    snli_dev_file = os.path.join(snli_dir, "snli_1.0_dev.txt")
    snli_test_file = os.path.join(snli_dir, "snli_1.0_test.txt")
    files = [snli_train_file, snli_dev_file, snli_test_file]
    data = {}
    data['train'] = preprocess(snli_train_file)
    data['dev'] = preprocess(snli_dev_file)
    data['test'] = preprocess(snli_test_file)
    
    return data    

def read_vectors(path, vocabulary_from_data):
    vectors = {}
    iw = []
    wi = {}
    iw.append('UNK')
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            tokens = line.rstrip().split(' ')
            if tokens[0] in vocabulary_from_data:
                vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                iw.append(tokens[0])
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi

def reformat_data(dataset, word2index):
    dataset['input_sent1'] = dataset.apply(lambda row: [word2index[x] if x in word2index else 0 for x in row['tokenized_sent1'] ], axis=1)
    dataset['input_sent2'] = dataset.apply(lambda row: [word2index[x] if x in word2index else 0 for x in row['tokenized_sent1'] ], axis=1)
    dataset = dataset[['input_sent1', 'input_sent2', 'gold_label']]
    return dataset

if __name__ == '__main__':
    home_dir = "../"
    data_dir = os.path.join(home_dir, "data")

    data = initialize(data_dir)

    vocabulary_from_data = build_vocabulary_from_data(data)
    emb_filename = os.path.join(data_dir, "glove.840B.300d.txt")
    # based on vocabulary from training data
    # build embedding matrix and dictionary using emb file
    vectors, iw, wi = read_vectors(emb_filename, vocabulary_from_data)
    emb_for_data_filename = os.path.join(data_dir, "emb_for_data") 
    pl.dump([vectors, iw, wi], open(emb_for_data_filename, "wb")) 
    # based on the dictionary, reformat the train/validation/test
    # dataset
    train = data['train']
    dev = data['dev']
    test = data['test']

    train = reformat_data(train, wi)
    dev = reformat_data(dev, wi)
    test = reformat_data(test, wi)
    train_file_name = os.path.join(data_dir, "train_file")
    dev_file_name = os.path.join(data_dir, "dev_file")
    test_file_name = os.path.join(data_dir, "test_file")
    train.to_pickle(train_file_name)
    dev.to_pickle(dev_file_name)
    test.to_pickle(test_file_name)

