import pickle as pl
import tensorflow as tf
import argparse
import os
import numpy as np
from components import DataIterator
from components import dan_as_encoder, bilstm_as_encoder

def build_graph(
    inputs1,
    inputs2,
    emb_matrix,
    encoder,
    embedding_size = 300,
    layer_size = None,
    nclasses = 3
    ):

    print(" input1 shape: "+str(inputs1.shape))
    print(" input2 shape: "+str(inputs2.shape))
    
  #  reuse_var = None 
   # reuse_encoder_var = None 
    word_embeddings = tf.convert_to_tensor(emb_matrix, np.float32)
    print("word_embeddings shape: "+str(word_embeddings.shape))
    print(word_embeddings)

    # the encoders
    with tf.variable_scope("encoder_vars") as encoder_scope:
        encoded_input1 = encoder(inputs1, word_embeddings, layer_size)
        encoder_scope.reuse_variables()
        encoded_input2 = encoder(inputs2, word_embeddings, layer_size)
        print("encoded inputs1 shape: "+str(encoded_input1.shape))
        print("encoded inputs2 shape: "+str(encoded_input2.shape))
        
    abs_diffed = tf.abs(tf.subtract(encoded_input1, encoded_input2))
    print(abs_diffed)
    
    multiplied = tf.multiply(encoded_input1, encoded_input2)
    print(multiplied)
    concatenated = tf.concat([encoded_input1, encoded_input2,
abs_diffed, multiplied], 1) 
    print(concatenated)
    concatenated_dim = concatenated.shape.as_list()[1]
    
    # the fully-connected dnn layer
    # fix it as 512
    fully_connected_layer_size = 512
    with tf.variable_scope("dnn_vars") as encoder_scope:
        wd = tf.get_variable(name="wd", dtype=tf.float32,
shape=[concatenated_dim, fully_connected_layer_size])
        bd = tf.get_variable(name="bd", dtype=tf.float32,
shape=[fully_connected_layer_size])
    dnned = tf.matmul(concatenated, wd) + bd
    print(dnned)
        
    with tf.variable_scope("out") as out:
        w_out = tf.get_variable(name="w_out", dtype=tf.float32,
shape=[fully_connected_layer_size, nclasses])
        b_out = tf.get_variable(name="b_out", dtype=tf.float32,
shape=[nclasses])
    logits = tf.matmul(dnned, w_out) + b_out
    
    return logits

def evaluate(evaluate_on_data, sess, batch_size, depth=3, on_value=1,
off_value=0):
    data = DataIterator(evaluate_on_data)
    total_batch = int(len(evaluate_on_data)/batch_size)
    avg_accu = 0.0
    for i in range(total_batch):
        batch_data_sent1, batch_data_sent2, batch_label = data.next_batch(batch_size)
        batch_label_formatted = tf.one_hot(indices=batch_label, depth=depth, on_value=on_value, off_value=off_value, axis=-1)
        batch_label_one_hot = sess.run(batch_label_formatted)
        feed = {inputs1_: batch_data_sent1, inputs2_: batch_data_sent2, labels_: batch_label_one_hot}
        accu  = sess.run(accuracy, feed_dict=feed)
        avg_accu += 1.0*accu/total_batch

    return avg_accu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for building the model.')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                   help='training batch size')
    parser.add_argument('-c', '--encoder', type=str, default="bilstm",
                   help='encoder type')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                   help='epochs for training')
    parser.add_argument('-v', '--encoded_dim', type=int,
default=512, help='embedding size for compressed vector')
    parser.add_argument('-n', '--optimization', type=str, default="sgd",
                   help='optimization algorithm')
    parser.add_argument('-r', '--resume', type=bool, default=False,
                   help='pick up the latest check point and resume')


    args = parser.parse_args()
    print("args: "+str(args))
    resume = args.resume
    train_batch_size = args.batch_size
    epochs = args.epochs
    encoder_type = args.encoder
    encoded_dim = args.encoded_dim
    optimization = args.optimization

    encoder = None
    layer_size = None
    if encoder_type == "dan":
        encoder = dan_as_encoder
        layer_size = [300, 300, encoded_dim]
    elif encoder_type == "bilstm":
        encoder = bilstm_as_encoder
        layer_size = [encoded_dim]
        

    tf.set_random_seed(1234)
    home_dir = "../"
    data_dir = os.path.join(home_dir, "data")
    log_dir = os.path.join(home_dir, "logs")

    save_data_filename = os.path.join(data_dir, "save_data")
    [train, dev, test, emb_matrix] = pl.load(open(save_data_filename,
"rb"))
    train_data = DataIterator(train)

    nclasses = 3 
    sent_length = 50   
    decay_factor = 0.99
    inputs1_ = tf.placeholder(tf.int32, [None, sent_length], name="inputs1_holder")
    inputs2_ = tf.placeholder(tf.int32, [None, sent_length], name="inputs2_holder")
    labels_ = tf.placeholder(tf.int32, [None, nclasses], name="labels_holder")
    print("labels:"+str(labels_))

    logits = build_graph(inputs1_, inputs2_, emb_matrix, encoder,
layer_size=layer_size)
    print("logits: "+str(logits))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits))
    print("cross entropy: "+str(cross_entropy))

    optimizer = None
    learning_rate = tf.placeholder(tf.float32, shape=[])
    with tf.variable_scope('optimizers'):
        if optimization == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
        elif optimization == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
         
        y_predict = tf.argmax(logits, 1)
#     print("y predict: "+str(y_predict))
        correct_prediction = tf.equal(y_predict, tf.argmax(labels_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        dev_batch_size = 4096 
        test_batch_size = 4096 
        total_train_batch = int(len(train)/(train_batch_size))
        total_dev_batch = int(len(dev)/(dev_batch_size))
     
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        resume_from_epoch = -1 
        if resume:
            latest_cpt_file = tf.train.latest_checkpoint('../log')
            print("the code pick up from lateset checkpoint file: {}".format(latest_cpt_file))
            resume_from_epoch = int(float(str(latest_cpt_file).split('/')[-1].split('-')[1]))
            print("it resumes from previous epoch of {}".format(resume_from_epoch))
            saver.restore(sess, latest_cpt_file)

        depth = 3
        on_value = 1
        off_value = 0

        last_dev_accu = 0
        adaptive_learning_rate = None
        if optimization == "adam":
            adaptive_learning_rate = 0.01
        elif optimization == "sgd":
            adaptive_learning_rate = 0.1

        for epoch in range(resume_from_epoch+1, resume_from_epoch+epochs+1):
            # training part
            avg_cost = 0.0
            print("in epoch {}".format(epoch))
            for i in range(total_train_batch):
                batch_data_sent1, batch_data_sent2, batch_label = train_data.next_batch(train_batch_size)
                batch_label_formatted = tf.one_hot(indices=batch_label, depth=depth, on_value=on_value, off_value=off_value, axis=-1)
                batch_label_one_hot = sess.run(batch_label_formatted)
                
                feed = {inputs1_: batch_data_sent1, inputs2_: batch_data_sent2, labels_: batch_label_one_hot, learning_rate: adaptive_learning_rate}
                _, cross_en = sess.run([optimizer, cross_entropy], feed_dict=feed)
                avg_cost += cross_en/total_train_batch
            print("avg cost in the training phase epoch {}: {}".format(epoch, avg_cost))
            adaptive_learning_rate = adaptive_learning_rate * decay_factor
            # save the model
            saver.save(sess, os.path.join(log_dir, "model.ckpt"), epoch, write_meta_graph=False) 
            # dev part
            dev_accu = evaluate(dev, sess, dev_batch_size) 
            print("dev accuracy on dev set is {}".format(dev_accu))
            if dev_accu <= last_dev_accu:
                adaptive_learning_rate = adaptive_learning_rate/5.0
                print("dev accuracy decreased, chane learning rate to {}".format(adaptive_learning_rate)) 

        print("evaluating...")
    
        test_accu = evaluate(test, sess, test_batch_size) 
        print("prediction accuracy on test set is {}".format(test_accu))
