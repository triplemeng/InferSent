import nltk
import pickle as pl
import tensorflow as tf
import argparse
import os
import numpy as np
from tensorflow.contrib.keras import preprocessing
from components import bilstm_as_encoder

def find_rep_using_dict(data, wi):
    return [wi[ele] if ele in data else 0 for ele in data]
def pad_rep(rep, sent_length=50):
    padded = preprocessing.sequence.pad_sequences([rep], maxlen=sent_length, padding="post", truncating="post",
value=0)
    return padded

if __name__ == "__main__":
    home_dir = "../"
    data_dir = os.path.join(home_dir, "data")
    emb_for_data_filename = os.path.join(data_dir, "emb_for_data") 
    vectors, iw, wi = pl.load(open(emb_for_data_filename, "rb")) 
    save_data_filename = os.path.join(data_dir, "save_data")
    [_, _, _, emb_matrix] = pl.load(open(save_data_filename, "rb"))
    word_embeddings = tf.convert_to_tensor(emb_matrix, np.float32)

    sentence1 = input("Enter your sentence: ") 
    
    data1 = nltk.word_tokenize(sentence1.lower())
    
    rep1 = find_rep_using_dict(data1, wi)
    print(rep1)
    padded_rep1 = pad_rep(rep1)
    latest_cpt_file = tf.train.latest_checkpoint('../logs')
    print("the code pick up from lateset checkpoint file: {}".format(latest_cpt_file))
    resume_from_epoch = int(str(latest_cpt_file).split('/')[-1].split('-')[1])
    print("it resumes from previous epoch of {}".format(resume_from_epoch))

    vector_embedding_size = 512
    sent_length = 50
    inputs1_ = tf.placeholder(tf.int32, [1, sent_length], name="inputs1_holder")
    with tf.variable_scope("encoder_vars") as encoder_scope:
        encoded1_ = bilstm_as_encoder(inputs1_, word_embeddings, layer_size=[vector_embedding_size])
        encoder_scope.reuse_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, latest_cpt_file)
        feed = {inputs1_: padded_rep1}
        encoded1= sess.run(encoded1_, feed_dict=feed)
        print("input sentence is: {}".format(sentence1))
        print(encoded1)
    


    pass
