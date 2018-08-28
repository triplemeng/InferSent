import pickle as pl
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.contrib.keras import preprocessing

def gen_label(row): 
    if str(row['gold_label'])=='contradiction':
        return 0
    elif str(row['gold_label'])=='neutral': 
        return 1
    else:
        return 2

def pad_data(data, func, sent_length = 50 ):
    data['gold_label'] = data.apply(func, axis = 1)
    padded1 = preprocessing.sequence.pad_sequences(data['input_sent1'].values.tolist(), maxlen=sent_length, padding="post", truncating="post",
value=0)
    data['input_sent1_padded'] = pd.Series(padded1.tolist())
    padded2 = preprocessing.sequence.pad_sequences(data['input_sent2'].values.tolist(), maxlen=sent_length, padding="post", truncating="post",
value=0)
    data['input_sent2_padded'] = pd.Series(padded2.tolist())
    data = data.sample(frac=1)

def dan_as_encoder(sent_padded_as_tensor, word_embeddings, layer_size):
    embed = tf.nn.embedding_lookup(word_embeddings,
sent_padded_as_tensor)
    print("embedded input shape: "+str(embed.shape))
    mean_embed = tf.reduce_mean(embed, 1)
    print("mean embedded input: "+str(mean_embed))
    print("mean embedded input shape: "+str(mean_embed.shape))

    embedding_size = mean_embed.shape.as_list()[1]
    
    w1 = tf.get_variable(name="w1", dtype=tf.float32,
shape=[embedding_size, layer_size[0]])
    b1 = tf.get_variable(name="b1", dtype=tf.float32,
shape=[layer_size[0]])
    r1 = tf.matmul(mean_embed, w1) + b1
    print("r1 shape: "+str(r1.shape))
    
    w2 = tf.get_variable(name="w2", dtype=tf.float32,
shape=[layer_size[0], layer_size[1]])
    b2 = tf.get_variable(name="b2", dtype=tf.float32,
shape=[layer_size[1]])
    r2 = tf.matmul(r1, w2) + b2
    print("r2 shape: "+str(r2.shape))
    
    w3 = tf.get_variable(name="w3", dtype=tf.float32,
shape=[layer_size[1], layer_size[2]])
    b3 = tf.get_variable(name="b3", dtype=tf.float32,
shape=[layer_size[2]])
    encoded = tf.matmul(r2, w3) + b3
    print("encoded shape: "+str(encoded.shape))
        
    return encoded

def bilstm_as_encoder(sent_padded_as_tensor, word_embeddings,
layer_size, hidden_size=100, sent_length=50, embedding_size=300):
    embed_input = tf.nn.embedding_lookup(word_embeddings,
sent_padded_as_tensor)
    print("sent_padded_as_tensor: "+str(sent_padded_as_tensor))
    print("embed_input: "+str(embed_input))

    cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size)
    print('build fw cell: '+str(cell_fw))
    cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size)
    print('build bw cell: '+str(cell_bw))

    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
inputs=embed_input, dtype=tf.float32)
    print('rnn outputs: '+str(rnn_outputs))   

    concatenated_rnn_outputs = tf.concat(rnn_outputs, 2)
    print('concatenated rnn outputs: '+str(concatenated_rnn_outputs))   
  
    max_pooled = tf.layers.max_pooling1d(concatenated_rnn_outputs,
sent_length, strides=1) 
    print('max_pooled: '+str(max_pooled))   

    max_pooled_formated = tf.reshape(max_pooled, [-1, 2*hidden_size])
    print('max_pooled_formated: '+str(max_pooled_formated))   
 
    w1 = tf.get_variable(name="w1", dtype=tf.float32,
shape=[2*hidden_size, layer_size[0]])
    b1 = tf.get_variable(name="b1", dtype=tf.float32,
shape=[layer_size[0]])
    encoded = tf.matmul(max_pooled_formated, w1) + b1
     
    return encoded

def build_emb_matrix(index2word, vectors, embedding_size=300):
    dict_size = len(index2word)
    emb_matrix = np.zeros((dict_size, embedding_size))
    for k in  range(1, dict_size):
        word = index2word[k]
        emb_matrix[k] = vectors[word]
    return emb_matrix


class DataIterator():
    def __init__(self, df):
       
        self.df = df
        self.size = int(len(df))
        self.cursor = 0
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe 
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()

        res = self.df.iloc[self.cursor:self.cursor+n]
        self.cursor += n
        return np.asarray(res['input_sent1_padded'].tolist()), np.asarray(res['input_sent2_padded'].tolist()), res['gold_label'].tolist()

if __name__ == '__main__':
    home_dir = "../"
    data_dir = os.path.join(home_dir, "data")
    emb_for_data_filename = os.path.join(data_dir, "emb_for_data")
    save_data_filename = os.path.join(data_dir, "save_data")
    vectors, index2word, word2index  = pl.load(open(emb_for_data_filename, "rb"))
    
    train_file_name = os.path.join(data_dir, "train_file")
    dev_file_name = os.path.join(data_dir, "dev_file")
    test_file_name = os.path.join(data_dir, "test_file")
    train = pd.read_pickle(train_file_name)
    dev = pd.read_pickle(dev_file_name)
    test = pd.read_pickle(test_file_name)
    pad_data(train, gen_label)
    pad_data(dev, gen_label)
    pad_data(test, gen_label)
    emb_matrix = build_emb_matrix(index2word, vectors)
    pl.dump([train, dev, test,
emb_matrix], open(save_data_filename,
"wb"))


    pass
